import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def getPredictor() -> Predictor:
    config = Cfg.load_config_from_name('vgg_transformer')

    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)
    return detector


def predictText(image, detector: Predictor) -> str:
    image = Image.fromarray(image)
    text = detector.predict(image, return_prob=False)
    return text


if __name__ == "__main__":

    config = Cfg.load_config_from_name('vgg_transformer')

    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)


    img = 'img/word_1.jpg'
    img = Image.open(img)
    plt.imshow(img)
    s = detector.predict(img)
    print(s)