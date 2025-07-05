

from training import(
    config,
    datasets,
    models,
)


# We need to import classes from each of these six files

# from training import(
#     classifier,
#     GAN,
#     SNGAN,
#     WGAN,
#     TMG_GAN,
#     logging,
# )

from training.logger import Logger
from training.TMG_GAN import TMGGAN
from training.GAN import GAN
from training.WGAN import WGAN
from training.SNGAN import SNGAN
from training.classifier import Classifier
from training.TMG_GAN_Dynamic import TMGGANDynamic
from training.TMG_GAN_DPL import MYMETHOD
