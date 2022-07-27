import cv2
import numpy as np
import os
import PySimpleGUI as sg


def getposHsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("HSV is", HSV[y, x])


def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def inside(circle, x, y):
    return distance(circle[0], circle[1], x, y) < circle[2]


def select_circle(x, y):
    global selected_circle, new_circle, update_image
    for index, circle in enumerate(circles):
        if inside(circle, x, y):
            selected_circle = index
            new_circle = True
            update_image = True
            return True
    selected_circle = -1
    return False


def create_new_circle(x, y):
    global circles
    circles = np.vstack([circles, [x, y, 50]])
    select_circle(x, y)
    pass

def delete_selected_circle():
    global selected_circle, circles, update_image
    if selected_circle != -1:
        circles = np.delete(circles, selected_circle, 0)
        update_image = True

def mouse_input_photo(event, x, y, flags, param):
    global update_image, rbutton_down, mbutton_down
    if event == cv2.EVENT_LBUTTONDOWN:
        if select_circle(x, y):
            return
        else:
            create_new_circle(x, y)

    if event == cv2.EVENT_RBUTTONDOWN or rbutton_down:
        rbutton_down = True
        if selected_circle != -1:
            circles[selected_circle] = [x, y, circles[selected_circle][2]]
            update_image = True
            select_circle(x, y)

    if event == cv2.EVENT_RBUTTONUP:
        rbutton_down = False

    if event == cv2.EVENT_MBUTTONDOWN or mbutton_down:
        mbutton_down = True
        if selected_circle != -1:
            circle_x = circles[selected_circle][0]
            circle_y = circles[selected_circle][1]
            r = distance(circles[selected_circle][0], circles[selected_circle][1], x, y)
            circles[selected_circle] = [circle_x, circle_y, r]
            update_image = True
            select_circle(circle_x, circle_y)

    if event == cv2.EVENT_MBUTTONUP:
        mbutton_down = False


def draw_circles_on_image(input_image):
    output = input_image.copy()
    for index, (x, y, r) in enumerate(circles):
        if index == selected_circle:
            cv2.circle(output, (x, y), r, (0, 0, 255), 1)
        else:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        # if(r*scale > 20.5 and r*scale < 26):
        #     cv2.putText(output, "5zl", org=(x,y), fontFace = 4, fontScale = 2, color = (255,255,0),thickness = 5, lineType = cv2.LINE_AA)
        sizes = map_of_coins.keys()
        size = min(sizes, key=lambda x: abs(x - (r * scale)))

        cv2.putText(output, map_of_coins.get(size), org=(x, y), fontFace=4, fontScale=2, color=(255, 255, 0),
                    thickness=5,
                    lineType=cv2.LINE_AA)

    return output


def load_circlkls(param):
    return np.load(param)


def save_circles(param):
    file1 = open(param + ".txt", "w")
    file1.write(str(scale))
    file1.close()
    np.save(param, circles)


def save_last_image(param):
    file1 = open("last.txt", "w")
    file1.write(param)
    file1.close()


def load_last_image():
    try:
        file1 = open("last.txt", "r")
        out = file1.readline()
        print(out)
        file1.close()
        if out == "":
            return None
        return out
    except:
        return None


def show_circles_window():
    global update_image, img, w, h, next_image, scale

    directory = os.fsencode(src)
    window = built_window(None, 0, 0, 0)
    last_circle = load_last_image()
    for file in os.listdir(directory):
        if last_circle is not None:
            if os.path.normpath(file).decode("utf-8") != last_circle:
                continue
            else:
                last_circle = None
        scale = 0
        path = "obrazy/" + (os.path.normpath(file).decode("utf-8"))
        print(path)
        img = cv2.imread(path, 1)
        w = img.shape[1]
        h = img.shape[0]
        img = cv2.resize(img, (w // 4, h // 4))
        output = img.copy()
        generate_circles_hsv()
        output = draw_circles_on_image(output)

        cv2.imshow("window", output)
        # cv2.moveWindow('window', 200, 200)
        # cv2.setMouseCallback("window",getposHsv)

        cv2.setMouseCallback("window", mouse_input_photo)

        k = 0
        while True:
            next_image = False
            cv2.imshow("window", output)

            upadte_window(window)
            if update_image:
                output = draw_circles_on_image(img)
            if next_image and scale != 0:
                save_circles("dane/" + (os.path.normpath(file).decode("utf-8")))
                save_last_image((os.path.normpath(file).decode("utf-8")))
                break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def built_window(event, x_circle, y_circle, r_circle):
    layout = [[sg.In(default_text=x_circle, key="X"),
               sg.In(default_text=y_circle, key="Y"),
               sg.In(default_text=r_circle, key="R"),
               sg.In(default_text=0, key="s")],
              [sg.Button("UPDATE"), sg.Button("OK"), sg.Button("scale"), sg.Button("Delete")]]
    return sg.Window("Demo", layout)


def upadte_window(window):
    global selected_circle, new_circle, update_image, next_image
    update_image = False  # todo
    event, values = window.read(timeout=10)

    if selected_circle != -1 and new_circle:
        new_circle = False
        window.Element("X").update(value=circles[selected_circle][0])
        window.Element("Y").update(value=circles[selected_circle][1])
        window.Element("R").update(value=circles[selected_circle][2])

    if event == "UPDATE":
        # showCircles(circles, img)
        circles[selected_circle] = [values["X"], values["Y"], values["R"]]
        print("==================||===============")
        print(circles)
        update_image = True
    if event == "OK":
        next_image = True

    if event == "scale":
        global scale
        real_size = map_of_values[int(values["s"])]
        r = circles[selected_circle][2]
        scale = real_size / r

    if event == "Delete":
        delete_selected_circle()

    if event == sg.WIN_CLOSED:
        if update_image:
            # todo save change

            pass
        exit(0)


def generate_circles_hsv():
    global HSV, circles
    work_image = img.copy()
    # cv2.setMouseCallback("window", getposHsv)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((9, 9), np.uint8) / 81
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # gray = cv2.GaussianBlur(gray, (11, 11), cv2.BORDER_DEFAULT)

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=70,minRadius=5 , maxRadius=100)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=70, minRadius=5, maxRadius=100)

    print(circles)

    circles = np.round(circles[0, :]).astype("int")


def main():
    global HSV, circles
    show_circles_window()


if __name__ == "__main__":
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

    src = "obrazy"
    img = None
    w = 0
    h = 0
    selected_circle = -1
    new_circle = False
    circles = None
    update_image = False
    rbutton_down = False
    mbutton_down = False
    next_image = False
    scale = 24.0 / 99.4
    HSV = None

    main()
