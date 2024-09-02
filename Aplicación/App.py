import sys
import os
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QPushButton, QVBoxLayout, QScrollArea, QWidget
from PyQt5.QtCore import Qt
from PIL import Image
import tensorflow as tf
import numpy as np
from keras.models import load_model
from efficientnet.tfkeras import EfficientNetB5

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class Skin_analizer(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(resource_path("gui_app.ui"), self)
        self.setWindowTitle("Skin Analyzer")
        self.setFixedSize(1120, 820)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setWindowIcon(QIcon(resource_path('AppImg/LogoIconApp.png')))
        self.modelo = load_model(resource_path('modelo.hdf5'))
        self.imgList = []
        self.initUI()

    def initUI(self):
        self.actualFrame = self.InfoFrame
        self.AnalizerFrame.hide()
        self.HistoryFrame.hide()
        self.AnalizerButton.clicked.connect(self.AnalizerButtonAction)
        self.HistoryButton.clicked.connect(self.HistoryButtonAction)
        self.InfoButton.clicked.connect(self.InfoButtonAction)
        self.initAnalizerFrame()

    def initAnalizerFrame(self):
        self.AnaImg.mousePressEvent = self.open_link
        self.AnaAnalizerButton.clicked.connect(self.cargar_y_predecir)

    def open_link(self, event):
        import webbrowser
        webbrowser.open('https://example.com')



    def AnalizerButtonAction(self):
        self.AnalizerFrame.show()
        self.HistoryFrame.hide()
        self.InfoFrame.hide()

    def HistoryButtonAction(self):
        self.AnalizerFrame.hide()
        self.updateHistory()
        self.HistoryFrame.show()
        self.InfoFrame.hide()

    def InfoButtonAction(self):
        self.AnalizerFrame.hide()
        self.HistoryFrame.hide()
        self.InfoFrame.show()

    def updateHistory(self):
        if self.imgList:
            for i, (filepath, diagnosis) in enumerate(reversed(self.imgList[-8:])):
                labelName = f"HisNameLabel_{i + 1}"
                labelImg = f"HisImgLabel_{i + 1}"
                labelDiag = f"HisDiagLabel_{i + 1}"
                labelNameCrea = getattr(self, labelName, None)
                labelImgCrea = getattr(self, labelImg, None)
                labelDiagCrea = getattr(self, labelDiag, None)
                if labelNameCrea:
                    labelNameCrea.setText(os.path.basename(filepath))
                if labelImgCrea:
                    labelImgCrea.setStyleSheet("border: 2px solid #A9A9A9; padding: 5px;")
                    labelImgCrea.setPixmap(QPixmap(filepath).scaled(300, 300))
                if labelDiagCrea:
                    if diagnosis < 0.5:
                        labelDiagCrea.setText("Benigno")
                        labelDiagCrea.setStyleSheet("background-color: #00fa14; border-radius: 10px; color: white;")
                    else:
                        labelDiagCrea.setText("Maligno")
                        labelDiagCrea.setStyleSheet("background-color: #f90101; border-radius: 10px; color: white;")

    def cargar_y_predecir(self):
        filepath, _ = QFileDialog.getOpenFileName(self, 'Seleccionar imagen', '', 'Imagen (*.png *.jpg *.jpeg)')
        if filepath:
            self.AnaImg.setPixmap(QPixmap(filepath))
            self.AnaPhotoIcon.hide()
            self.AnaImgNameLabel.setText(os.path.basename(filepath))
            image = Image.open(filepath)
            image = image.resize((512, 512))
            image = np.array(image) / 255.0
            tensor = tf.expand_dims(image, 0)
            diagnosis = self.modelo.predict(tensor)[0]
            self.appendNoDup((filepath, diagnosis), self.imgList)
            self.show_diagnosis(diagnosis)

    def show_diagnosis(self, diagnosis):
        if diagnosis < 0.5:
            self.AnaDiagLabel.setText("Benigno")
            self.AnaDiagLabel.setStyleSheet("background-color: #00fa14; border-radius: 10px; color: white;")
        else:
            self.AnaDiagLabel.setText("Maligno")
            self.AnaDiagLabel.setStyleSheet("background-color: #f90101; border-radius: 10px; color: white;")

    def appendNoDup(self, archivo, lista):
        if archivo in lista:
            lista.append(lista.pop(lista.index(archivo)))
        else:
            lista.append(archivo)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("AppImg/LogoIcon.png")))
    GUI = Skin_analizer()
    GUI.show()
    sys.exit(app.exec_())
