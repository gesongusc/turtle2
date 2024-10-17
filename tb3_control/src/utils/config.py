import os
import shutil
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
import datetime
import time
from easydict import EasyDict
from pprint import pprint
import time
from utils.dirs import create_dirs


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    if not main_logger.handlers:
        main_logger.addHandler(console_handler)
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(args):
    config, _ = get_config_from_json(args.config)

    config.mode = args.mode
    config.num_agents = args.num_agents
    config.map_w = args.map_w
    config.map_h = args.map_w
    config.map_density = args.map_density
    config.map_type = args.map_type

    config.trained_num_agents = args.trained_num_agents
    config.trained_map_w = args.trained_map_w
    config.trained_map_h = args.trained_map_w
    config.trained_map_density = args.trained_map_density
    config.trained_map_type = args.trained_map_type

    config.nGraphFilterTaps = args.nGraphFilterTaps

    config.num_testset = args.num_testset
    config.load_num_validset = args.load_num_validset
    config.num_validset = args.load_num_validset

    config.con_train = args.con_train
    config.lastest_epoch = args.lastest_epoch
    config.best_epoch = args.best_epoch
    config.test_general = args.test_general
    config.test_epoch = args.test_epoch
    config.log_anime = args.log_anime
    config.rate_maxstep = args.rate_maxstep

    config.vary_ComR_FOV = args.vary_ComR_FOV
    config.commR = args.commR
    config.dynamic_commR = args.dynamic_commR
    config.symmetric_norm = args.symmetric_norm

    config.guidance = args.guidance

    config.id_env = args.id_env
    config.action_select = args.action_select

    config.data_set = args.data_set
    config.nAttentionHeads = args.nAttentionHeads
    config.AttentionConcat = args.AttentionConcat

    config.tb_ExpName = args.tb_ExpName
    config.use_dropout = args.use_dropout
    config.batch_numAgent = args.batch_numAgent
    config.GSO_mode = args.GSO_mode
    config.attentionMode = args.attentionMode

    config.GNNGAT = args.GNNGAT
    config.CNN_mode = args.CNN_mode

    config.no_ReLU = args.no_ReLU
    config.test_on_ValidSet = args.test_on_ValidSet
    config.shuffle_testSet = args.shuffle_testSet

    config.default_actionSelect = args.default_actionSelect
    config.extra_policy = args.extra_policy

    config.numInputFeatures = args.numInputFeatures
    config.numInputFeaturesAP = args.numInputFeaturesAP


    if args.vary_ComR_FOV:
        config.data_root = os.path.join(config.data_root, "ComR_{}_Rv_{}".format(config.commR, int(config.FOV / 2)))

    if config.mode == 'train':
        if config.con_train:  # 断点续训
            config.exp_time_load = args.log_time_trained
            config.exp_time = config.exp_time_load
        else:
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple())))
    elif config.mode == "test":
        config.exp_time_load = args.log_time_trained + "/"
        config.exp_time = args.log_time_trained

    env_Setup = "{}{}x{}_rho{}_{}Agent".format(config.map_type, config.map_w, config.map_w, config.map_density, config.trained_num_agents)

    if config.exp_net == 'dcpOEGAT':
        config.exp_hyperPara = "K{}_P{}".format(config.nGraphFilterTaps, config.nAttentionHeads)
    else:
        config.exp_hyperPara = "K{}".format(config.nGraphFilterTaps)

    if config.con_train:
        config.exp_name_load = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time_load)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name_load, "checkpoints/")
        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time)
        config.tb_exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.tb_ExpName, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints")
    elif config.test_general:
        env_Setup_load = "{}{}x{}_rho{}_{}Agent".format(config.trained_map_type, config.trained_map_w, config.trained_map_w,
                                                        config.trained_map_density, config.trained_num_agents)
        env_Setup_test = "{}{}x{}_rho{}_{}Agent".format(config.map_type, config.map_w, config.map_w,
                                                        config.map_density, config.num_agents)
        config.exp_name_load = os.path.join("{}_{}".format(config.exp_net, env_Setup_load), config.exp_hyperPara, config.exp_time_load)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name_load, "checkpoints/")
        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup_test), config.exp_hyperPara, config.exp_time)
        config.tb_exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup_test), config.exp_hyperPara, config.tb_ExpName, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints")
    else:
        config.exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.exp_time)
        config.tb_exp_name = os.path.join("{}_{}".format(config.exp_net, env_Setup), config.exp_hyperPara, config.tb_ExpName, config.exp_time)
        config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints/")

    config.out_dir = os.path.join(config.save_data, "experiments", config.exp_name, "out/")
    config.log_dir = os.path.join(config.save_data, "experiments", config.exp_name, "logs/")
    config.failCases_dir = os.path.join(config.save_data, "experiments", config.exp_name, "failure_cases/")

    create_dirs([config.checkpoint_dir, config.out_dir, config.log_dir, config.failCases_dir])

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("The configurations are successfully processed.")

    return config
