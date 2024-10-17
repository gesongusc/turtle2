from graphs.weights_initializer import weights_init
import utils.graphUtils.graphML as gml
from graphs.models.resnet_pytorch import *


class DecentralPlannerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        inW = 11
        inH = 11
        convW = [inW]
        convH = [inH]
        numAction = 5
        numChannel = [3] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]
        dimCompressMLP = 1
        numCompressFeatures = [self.config.numInputFeatures]
        
        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2
        dimNodeSignals = [self.config.numInputFeatures]
        nGraphFilterTaps = [self.config.nGraphFilterTaps]

        numFeatures_agentPath = (self.config.nearest_robot_num + 1) * self.config.nearest_step_num * 2
        dimCompressMLP_agentPath = 3
        numCompressFeatures_agentPath = [128, 64, 32]
        numCompressFeatures_agentPath = [numFeatures_agentPath] + numCompressFeatures_agentPath
        compressmlp_agentPath = []
        for l in range(dimCompressMLP_agentPath):
            compressmlp_agentPath.append(nn.Linear(in_features=numCompressFeatures_agentPath[l], out_features=numCompressFeatures_agentPath[l + 1], bias=True))
            compressmlp_agentPath.append(nn.ReLU(inplace=True))
        self.compressMLP_agentPath = nn.Sequential(*compressmlp_agentPath)

        # --- actionMLP
        if self.config.use_dropout:
            dimActionMLP = 2
            numActionFeatures = [self.config.numInputFeatures, numAction]
        else:
            dimActionMLP = 1
            numActionFeatures = [numAction]

        # CNN to extract feature
        if self.config.CNN_mode == 'ResNetSlim_withMLP':
            convl = []
            convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
            convl.append(nn.Dropout(0.2))
            convl.append(nn.Flatten())
            convl.append(nn.Linear(in_features=1152, out_features=self.config.numInputFeatures, bias=True))
            self.ConvLayers = nn.Sequential(*convl)
            numFeatureMap = self.config.numInputFeatures
        elif self.config.CNN_mode == 'ResNetLarge_withMLP':
            convl = []
            convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
            convl.append(nn.Dropout(0.2))
            convl.append(nn.Flatten())
            convl.append(nn.Linear(in_features=1152, out_features=self.config.numInputFeatures, bias=True))
            self.ConvLayers = nn.Sequential(*convl)
            numFeatureMap = self.config.numInputFeatures
        elif self.config.CNN_mode == 'ResNetSlim':
            convl = []
            convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
            convl.append(nn.Dropout(0.2))
            self.ConvLayers = nn.Sequential(*convl)
            numFeatureMap = 1152
        elif self.config.CNN_mode == 'ResNetLarge':
            convl = []
            convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
            convl.append(nn.Dropout(0.2))
            self.ConvLayers = nn.Sequential(*convl)
            numFeatureMap = 1152
        else:
            convl = []
            numConv = len(numChannel) - 1
            nFilterTaps = [3] * numConv
            nPaddingSzie = [1] * numConv
            for i in range(numConv):
                convl.append(nn.Conv2d(in_channels=numChannel[i], out_channels=numChannel[i + 1], kernel_size=nFilterTaps[i], stride=numStride[i], padding=nPaddingSzie[i], bias=True))
                convl.append(nn.BatchNorm2d(num_features=numChannel[i + 1]))
                convl.append(nn.ReLU(inplace=True))

                W_tmp = int((convW[i] - nFilterTaps[i] + 2 * nPaddingSzie[i]) / numStride[i]) + 1
                H_tmp = int((convH[i] - nFilterTaps[i] + 2 * nPaddingSzie[i]) / numStride[i]) + 1

                # Adding maxpooling http://cs231n.github.io/convolutional-networks/
                if i % 2 == 0:
                    convl.append(nn.MaxPool2d(kernel_size=2))
                    W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                    H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1

                convW.append(W_tmp)
                convH.append(H_tmp)

            self.ConvLayers = nn.Sequential(*convl)
            numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

        # MLP-feature compression
        numCompressFeatures = [numFeatureMap] + numCompressFeatures
        compressmlp = []
        for i in range(dimCompressMLP):
            compressmlp.append(nn.Linear(in_features=numCompressFeatures[i], out_features=numCompressFeatures[i + 1], bias=True))
            compressmlp.append(nn.ReLU(inplace=True))
        self.compressMLP = nn.Sequential(*compressmlp)
        self.numFeatures2Share = numCompressFeatures[-1]
        self.numFeatures2ShareAP = self.config.numInputFeaturesAP

        # graph neural network
        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]+numCompressFeatures_agentPath[-1]] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        # Graph Filtering Layers
        gfl = []
        for i in range(self.L):
            gfl.append(gml.GraphFilterBatch(self.F[i], self.F[i + 1], self.K[i], self.E, self.bias))
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # Nonlinearity
            if not self.config.no_ReLU:
                gfl.append(nn.ReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)

        # MLP --- map to actions
        numActionFeatures = [self.F[-1]] + numActionFeatures
        actionsfc = []
        for i in range(dimActionMLP):
            if i < (dimActionMLP - i):
                actionsfc.append(nn.Linear(in_features=numActionFeatures[i], out_features=numActionFeatures[i + 1], bias=True))
                actionsfc.append(nn.ReLU(inplace=True))
            else:
                actionsfc.append(nn.Linear(in_features=numActionFeatures[i], out_features=numActionFeatures[i + 1], bias=True))

            if self.config.use_dropout:
                actionsfc.append(nn.Dropout(p=0.2))
                print('Dropout is add on MLP')

        self.actionsMLP = nn.Sequential(*actionsfc)
        self.apply(weights_init)

    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S
        # Remove nan data
        self.S[torch.isnan(self.S)] = 0
        if self.config.GSO_mode == 'dist_GSO_one':
            self.S[self.S > 0] = 1
        elif self.config.GSO_mode == 'full_GSO':
            self.S = torch.ones_like(self.S).to(self.config.device)
        # self.S[self.S > 0] = 1

    def forward(self, inputTensor, inputTensorAP):


        (B, N, C, W, H) = inputTensor.shape
        input_currentAgent = inputTensor.reshape(B * N, C, W, H).to(self.config.device)
        featureMap = self.ConvLayers(input_currentAgent).to(self.config.device)
        featureMapFlatten = featureMap.view(featureMap.size(0), -1).to(self.config.device)
        compressfeature = self.compressMLP(featureMapFlatten).to(self.config.device)
        extractFeatureMap_old = compressfeature.reshape(B, N, self.numFeatures2Share).to(self.config.device)
        # extractFeatureMap = extractFeatureMap_old.permute([0, 2, 1]).to(self.config.device)
        

        (B_AP, N_AP, F_AP) = inputTensorAP.shape
        input_currentAgentAP = inputTensorAP.reshape(B_AP * N_AP, F_AP).to(self.config.device)
        compressfeatureAP = self.compressMLP_agentPath(input_currentAgentAP).to(self.config.device)
        extractFeatureMap_oldAP = compressfeatureAP.reshape(B_AP, N_AP, self.numFeatures2ShareAP).to(self.config.device)

        combineFeatureOld = torch.cat([extractFeatureMap_old, extractFeatureMap_oldAP], dim=2)
        combineFeature = combineFeatureOld.permute([0, 2, 1]).to(self.config.device)

        # DCP
        for l in range(self.L):
            self.GFL[2 * l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(combineFeature)

        (_, num_G, _) = sharedFeature.shape

        sharedFeature_permute = sharedFeature.permute([0, 2, 1]).to(self.config.device)
        sharedFeature_stack = sharedFeature_permute.reshape(B * N, num_G)

        action_predict = self.actionsMLP(sharedFeature_stack)

        return action_predict
