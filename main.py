from cv2 import Mat
from requests import get
from requests.exceptions import MissingSchema
from bs4 import BeautifulSoup, SoupStrainer
from PIL import Image
from pipe import where, select
from io import BytesIO
from collections import namedtuple
from nptyping import NDArray, Shape, Int, Float
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np
import cv2


WindowPos = namedtuple('WindowPos', 'left top right bottom width height hwnd')
Matches = namedtuple('Matches', 'tolerance, loc')

def url_validate(url: str) -> get:
    try:
        _img = get(url)
    except MissingSchema:
        return None
    except Exception:
        return None
    else:
        return _img


@dataclass
class ImgVars:
    """Image search locations and related data."""
    name: str
    x: int
    y: int
    tolerance: float
    match_result: NDArray[Shape['2, 2'], Float] # cv2.matchTemplate Length x Width array with match tolerance in each.
    to_search_img: NDArray[Shape['3, 3'], Int] # numpy array size (length, width, colors-BGR)


class ImageSearch:

    def __init__(self, _name: str, _c_wp: WindowPos, _to_search_img: np.array, _template: str, _mask: str, _threshold):
        self.name = _name
        self.to_search_img = _to_search_img # cv2.imdecode() image.
        # cv2.imdecode() uses height:width positing. Numpy slice with [top, left: bottom, right].
        a = self.to_search_img[_c_wp.top:_c_wp.bottom, _c_wp.left:_c_wp.right, 0:]
        self.capture = a.copy()
        self.template = cv2.imread(_template)
        if _mask != None:
            self.mask = cv2.imread(_mask)
        else:
            self.mask = None
        self.match_result = None
        self.threshold = _threshold
        self.matches = []

    def template_match(self):
        self.match_result = cv2.matchTemplate(self.capture, self.template, cv2.TM_CCOEFF_NORMED, mask=self.mask)
        height, width, colors = self.template.shape
        loc = np.where(self.match_result >= self.threshold)
        for pt in zip(*loc[::-1]):
            abc = self.match_result[pt[1], pt[0]]
            if abc != float("inf"):
                self.matches.append(Matches(abc, pt))
                # print("loc: {loc}, TM_CCOEFF_NORMED -- {result}".format(loc=pt, result=abc))
                top_left = pt
                _right_bottom = (top_left[0] + width, top_left[1] + height)
                cv2.rectangle(self.capture, top_left, _right_bottom, (255, 0, 0), 2)
        
        show_image(self.template, self.capture)

        if len(self.matches) > 0:
            self.matches = sorted(self.matches, key=lambda x: getattr(x, "tolerance"), reverse=True)

    def get_best_result(self):
        if len(self.matches) > 0:
            _iv = ImgVars(self.name, self.matches[0][1][0], self.matches[0][1][1], self.matches[0][0], 
            self.match_result, self.to_search_img)
            return _iv
        return None


def show_image(res, img):
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()


# Get the HTML for user supplied address.
html_request = url_validate('https://ibb.co/VQsFtZ9')

'''only_link_tags = SoupStrainer("link")
only_source = SoupStrainer(rel="image_src")
soup = BeautifulSoup(html_request.text, "html.parser", parse_only=only_source)
print(soup.prettify())'''

# make a list of HTML content with img tag
img_gen = [item for item in BeautifulSoup(html_request.text, "html.parser", parse_only=SoupStrainer(rel="image_src"))]

# filter img-tag list to remove invalid HTML address
path_gen = list(img_gen 
            | select( lambda x: url_validate(x['href'])) 
            | where( lambda x: x != None))
trade_img_res = None
original_img = None
for im in path_gen:
    original_img = cv2.imdecode(np.frombuffer(im.content, dtype=np.uint8), flags=cv2.IMREAD_COLOR )
    capture = WindowPos(0, 0, original_img.shape[1], original_img.shape[0], original_img.shape[1], original_img.shape[0], None)
    img = ImageSearch("scrape", capture, original_img , f"./Trading_with.png", None, 0.90)
    img.template_match()
    trade_img_res = img.get_best_result()
    if trade_img_res != None:
        img = None
        break
l = trade_img_res.x
t = trade_img_res.y - 10
r = original_img.shape[1]
b = trade_img_res.y + 50
capture = WindowPos(l, t, r, b, r - l, b - t, None)
img1 = ImageSearch("x close", capture, original_img, f"./closeX.png", None, 0.50)
img1.template_match()
x_img_res = img1.get_best_result()


a = 0
