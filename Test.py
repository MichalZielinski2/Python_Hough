import os
from random import random

import cv2
import numpy as np
import math


data_dir = "dane"
photos_dir = "obrazy"
data_size = None

confusion_matrix = None

map_on_numbers = {
    "1gr": 0,
    "2gr": 1,
    "5gr": 2,
    "10gr": 3,
    "20gr": 4,
    "50gr": 5,
    "1zl": 6,
    "2zl": 7,
    "5zl": 8
}

map_of_coins = {
    15.5: "1gr",
    17.5: "2gr",
    19.5: "5gr",
    16.5: "10gr",
    18.5: "20gr",
    20.5: "50gr",
    23.0: "1zl",
    21.5: "2zl",
    24.0: "5zl"
}
map_of_values = {
    1: 15.5,
    2: 17.5,
    5: 19.5,
    10: 16.5,
    20: 18.5,
    50: 20.5,
    100: 23.0,
    200: 21.5,
    500: 24.0
}

class hough_circles:
    def __init__(self, threshold, temp):
        self.threshold = threshold
        self.param1 = 30
        self.param2 = 70
        self.temperature = temp
        self.step = 0.99
        self.bestAcc = 0

    def generate_probability(self):
        self.temperature = self.temperature*self.step
        return self.temperature

    def train(self, train_data):
        #algorytm symulowanego wyżarzania.
        current_best_acc = 0;
        current_best_p1 = self.param1
        current_best_p2 = self.param2

        for i in range(100):   #todo range set

            tested_p1 = None
            tested_p2 = None

            best_of_four = current_best_acc
            bof_p1 = current_best_p1
            bof_p2 = current_best_p2


            tested_p1 = current_best_p1-1
            tested_p2 = current_best_p2
            acc = self.evaluate_parameters(tested_p1, tested_p2, train_data)
            if best_of_four < acc:
                best_of_four = acc
                bof_p1 = tested_p1
                bof_p2 = tested_p2


            tested_p1 = current_best_p1+1
            tested_p2 = current_best_p2
            acc = self.evaluate_parameters(tested_p1, tested_p2, train_data)
            if best_of_four < acc:
                best_of_four = acc
                bof_p1 = tested_p1
                bof_p2 = tested_p2


            tested_p1 = current_best_p1
            tested_p2 = current_best_p2 - 1
            acc = self.evaluate_parameters(tested_p1, tested_p2, train_data)
            if best_of_four < acc:
                best_of_four = acc
                bof_p1 = tested_p1
                bof_p2 = tested_p2


            tested_p1 = current_best_p1
            tested_p2 = current_best_p2 + 1
            acc = self.evaluate_parameters(tested_p1, tested_p2, train_data)
            if best_of_four < acc:
                best_of_four = acc
                bof_p1 = tested_p1
                bof_p2 = tested_p2

            current_best_acc = best_of_four
            current_best_p1 = bof_p1
            current_best_p2 = bof_p2

            if self.bestAcc < current_best_acc:
                self.param1 = current_best_p1
                self.param2 = current_best_p2
                self.bestAcc = current_best_acc
            else:
                return

        
    def evaluate(self, test_values):
        return self.evaluate_parameters(parameter_1=self.param1, parameter_2=self.param2, test_data=test_values)
       
    def evaluate_parameters (self, parameter_1, parameter_2, test_data):
        tp = 0
        fp = 0
        fn = 0
        
        for name, label in test_data.items():
            image = self.load_image(name)
            temp_tp, temp_fp, temp_fn = self.rate_image(parameter_1, parameter_2, image, label)
            
            tp += temp_tp
            fp += temp_fp
            fn += temp_fn
            
        return tp / (tp + fp + fn)



    def load_image_hq(self, name):
        path = photos_dir + "/" + name
        img = cv2.imread(path, 1)

        w = img.shape[1]
        h = img.shape[0]
        img = cv2.resize(img, (w // 4, h // 4))

        return img

    def load_image(self, name):
        path = photos_dir+"/"+name
        img = cv2.imread(path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        w = img.shape[1]
        h = img.shape[0]
        img = cv2.resize(img, (w // 4, h // 4))

        kernel = np.ones((9, 9), np.uint8) / 81
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        #todo szary, rozmycie
        return img

    def find_nearest(self, label, generated_circle):

        distance = 9999999.9
        circle = None

        for tmp_circle in label[0]:
            tmp_distance = self.distance(tmp_circle, generated_circle) #
            if tmp_distance < distance:
                distance = tmp_distance
                circle = tmp_circle
        return circle

    def rate_image(self, parameter_1, parameter_2, image, label):
        generated_circles = self.predict_parameters(image, parameter_1, parameter_2)
        
        tp = 0
        fp = 0
        fn = 0
        for generated_circle in generated_circles[0]:
            nearest = self.find_nearest(label, generated_circle)

            good = self.rate_circle(generated_circle, nearest)
            if good:
                tp += 1
            else:
                fp += 1
            
        return tp, fp, len(label[0]) - tp #todo
        
    def rate_circle(self, generated_circle, nearest):
        if self.fraction(generated_circle, nearest) > self.threshold:
            return True
        else:
            return False

    def predict_parameters(self, image, param1, param2):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=param1, param2=param2, minRadius=5, maxRadius=100)
        return circles

    def predict(self, image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=self.param1, param2=self.param2, minRadius=5, maxRadius=100)
        return circles
        
    def distance(self, circle1, circle2):
        return math.sqrt( (circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
        
    def fraction(self, circle1, circle2):
        def area(circle):
            return circle[2] ** 2 * math.pi
            
        distance = self.distance(circle1, circle2)
        if distance > circle1[2] or distance > circle2[2]: #sirodek poza
            return 0.0 
        elif (distance + circle1[2] > circle2[2]) != (distance + circle2[2] > circle1[2]): #wewnątcz
            circle1_area = area(circle1)
            circle2_area = area(circle2)
            if circle1_area > circle2_area:
                return circle2_area / circle1_area
            else:
                return circle1_area / circle2_area
        else: #przecinające się

            if circle1[2] < circle2[2]:
                tmp_circle = circle1
                circle1 = circle2
                circle2 = tmp_circle

            r1 = circle1[2]
            r2 = circle2[2]
            d = distance

            # 2*alpha = omega
            cos_omega = (r1**2 + d**2 - r2**2)/(2*d*r1)
            alpha = 2*math.acos(cos_omega)

            # 2*beta = delta
            cos_delta = (r2**2 + d**2 - r1**2) / (2*d*r2)
            beta = 2.0*(math.pi - math.acos(cos_delta))

            # pole łuków
            ar1 = alpha / (2*math.pi) * area(circle1)
            ar2 = beta / (2*math.pi) * area(circle2)
            # pole czworokąta
            p_rec = r1*d * math.sin(alpha/2.0)
            # mały baz soczewki
            mbs = ar1 - p_rec

            # pole soczewki
            p_lance = ar2 - mbs
            ret = (area(circle2) - p_lance) / (area(circle1) + p_lance)
            return ret
            # old
            # cos_omega = (circle1[2]**2 - distance**2 - circle2[2]**2) / (-2*distance*circle2[2])
            # omega = math.acos(cos_omega)
            #
            # cos_delta = (circle2[2]**2 - distance**2 - circle1[2]**2) / (-2*distance*circle1[2])
            # delta = math.acos(cos_delta)
            #
            # circle1_area = area(circle1)
            # circle2_area = area(circle2)
            #
            # circle1_segment_area = omega/math.pi * circle1_area
            # circle2_segment_area = delta/math.pi * circle1_area
            #
            # quadrangle_area = distance * circle1[2] * math.sin(delta)
            #
            # common_area = circle1_segment_area + circle2_segment_area - quadrangle_area
            #
            # ret = common_area / (circle1_area + circle2_area - common_area)
            # return ret
            
        return 0.0

    def show(self, labels):
        for name, label in labels.items():
            imagep = self.load_image(name)
            image = self.load_image_hq(name)
            predicted = self.predict(imagep)
            image = draw_circles_on_image(image, (255,0,0), predicted[0])
            image = draw_circles_on_image(image, (0,255,0), label[0])
            cv2.imshow("window", image)
            cv2.waitKey()

    def set_threshold(self, threshold):
        self.threshold = threshold

    def evaluate_circles(self, test):
        global confusion_matrix
        good = 0
        bad = 0
        confusion_matrix = [[0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0]]
        for name, label in test.items():
            image = self.load_image(name)
            circles = self.predict(image)
            tp_good, tp_bad = self.rate_coins(circles, label, label[1]) #

            good += tp_good
            bad += tp_bad
        if (good+bad) == 0:
            return "Undefined"
        return good/(good+bad)

    def rate_coins(self, circles, label, scale):
        good = 0
        bad = 0
        for circle in circles[0]:
            closest = self.find_nearest(generated_circle=circle, label=label) #
            if self.rate_circle(circle,closest):
                predicted = self.predict_class(circle, scale)
                true_class = self.predict_class(closest, scale)
                confusion_matrix[map_on_numbers[true_class]][map_on_numbers[predicted]] += 1
                if predicted == true_class:
                    good += 1
                else:
                    bad += 1

        return good, bad

    def predict_class(self, circle, scale):
        r = circle[2]
        sizes = map_of_coins.keys()
        size = min(sizes, key=lambda x: abs(x - (r * scale)))
        return map_of_coins.get(size)

def load_labels(data_dir):
    labels = {}
    for name in os.listdir(data_dir):
        if name.endswith(".npy"):
            circles = np.load(data_dir + "/" + name)
            f = open(data_dir + "/" + (name.replace(".npy", ".txt")), "r")
            scale = float(f.read())
            label = (circles,  scale)
            labels[name.replace(".npy", "")] = label

    return labels


def split_dic(dic, proc):
    d = math.floor(len(dic)*proc)
    iterator = 0;
    dic1 = {}
    dic2 = {}
    # items = dic.items()
    # dic1 = dict(items[d:])
    # dic2 = dict(items[:d])

    for key, item in dic.items():
        if iterator < d:
            dic1[key] = item
            iterator += 1
        else:
            dic2[key] = item

    return dic1, dic2

def draw_circles_on_image(input_image, color, circles):
    output = input_image.copy()
    for circle in circles:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        cv2.circle(output, (x, y), r, color, 1)
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

    return output


def training(train, test, threshold):
    model = hough_circles(threshold, 100)
    # model.train(train)
    model.param1 = 29
    model.param2 = 70
    score = model.evaluate(test)
    format_float = "{:.2f}".format(threshold)
    print("=====================||======================")

    print("threshold: ", format_float," :")
    print("\t param1:  \t",model.param1)
    print("\t param2:  \t", model.param2)
    print("\t detection:  \t", score)
    score = model.evaluate_circles(test)
    print("\t recognition:\t", score)
    print("\t matrix:\t")
    for line in confusion_matrix:
        print(line)



if __name__ == '__main__':
    labels = load_labels(data_dir)

    train, test = split_dic(labels, 0.00)

    training(train, test, 1.0)
    training(train, test, 0.99)
    training(train, test, 0.98)
    training(train, test, 0.97)
    training(train, test, 0.95)
    training(train, test, 0.93)
    training(train, test, 0.90)
    training(train, test, 0.85)
    training(train, test, 0.80)
    training(train, test, 0.75)


