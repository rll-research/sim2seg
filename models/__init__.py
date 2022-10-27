"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel

# Currently, only one model is provided.
# These can be updated to be different models.
CKPT_MEADOW_D = "PATH_TO_CKPT"
HPS_MEADOW_D = "--output_nc 6 --model pix2segdepth --ngf 128,256,256,512,512,512,1024"
CKPT_CANYON_D = CKPT_MEADOW_D
HPS_CANYON_D = HPS_MEADOW_D
CKPT_RL_D = CKPT_MEADOW_D
HPS_RL_D = HPS_MEADOW_D

CKPT_MEADOW = CKPT_MEADOW_D
HPS_MEADOW = HPS_MEADOW_D
CKPT_CANYON = CKPT_MEADOW_D
HPS_CANYON = HPS_MEADOW_D
CKPT_RL = CKPT_MEADOW_D
HPS_RL = HPS_MEADOW_D

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

def get_model_hps(model_type):
    # Just run with depth by default no matter what
    if model_type == "meadow":
        ckpt = CKPT_MEADOW
        hps = HPS_MEADOW
    elif model_type == "canyon":
        ckpt = CKPT_CANYON
        hps = HPS_CANYON
    elif model_type == "rl":
        ckpt = CKPT_RL
        hps = HPS_RL
    elif model_type == "meadow_depth":
        ckpt = CKPT_MEADOW_D
        hps = HPS_MEADOW_D
    elif model_type == "canyon_depth":
        ckpt = CKPT_CANYON_D
        hps = HPS_CANYON_D
    elif model_type == "rl_depth":
        ckpt = CKPT_RL_D
        hps = HPS_RL_D
    elif model_type == "meadow_transformer":
        ckpt = CKPT_ALL_T
        hps = HPS_MEADOW_T 
    else:
        raise NotImplementedError
    return ckpt, hps

def load_model(model_type, device_id):
    from options.base_options_simple import BaseOptionsSimple
    ckpt_path, hps = get_model_hps(model_type)
    opt = BaseOptionsSimple().parse(hps.split(" "))
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gpu_ids = [device_id]
    s2s_model= create_model(opt)
    s2s_model.custom_load(ckpt_path)
    s2s_model.device_id = device_id
    return s2s_model
