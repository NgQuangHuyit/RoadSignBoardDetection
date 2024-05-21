import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def getPredictor() -> Predictor:
    """ 
    Khởi tạo mô hình nhận dạng văn bản
    """
    config = Cfg.load_config_from_name('vgg_transformer')

    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)
    return detector


def predictText(image, detector: Predictor) -> str:
    """
    Thực hiện nhận dạng văn bản từ ảnh
    """
    image = Image.fromarray(image)
    text = detector.predict(image, return_prob=False)
    return text

