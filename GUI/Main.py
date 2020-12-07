import sys, random
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtChart import QChart, QChartView, QValueAxis, QBarCategoryAxis, QBarSet, QBarSeries, QHorizontalBarSeries, QPieSeries, QPieSlice, QAbstractBarSeries
from PyQt5.Qt import Qt
from connect_db import Grafik_2, Grafik, NamaPantai
from PyQt5 import QtCore, QtGui, QtWidgets
from fungsiKlasifikasi import Classification
from Details import Ui_Detail
import json
import numpy as np
import math


class Ui_MainWindow(QtWidgets.QMainWindow):

    def openDetails(self):
        
        self.window = QtWidgets.QDialog()
        self.ui = Ui_Detail()
        self.ui.setupUi(self.window)
        self.window.show()
        if self.inputUlasan.toPlainText()!='' :
            self.ui.setDetail(self.tmp_res[0],self.tmp_res[1],self.tmp_res[2],self.tmp_res[3],self.tmp_res[4],self.tmp_res[5])
            return

    def __init__(self, parent=None):       
        super().__init__(parent)
        self.setObjectName("MainWindow")
        self.resize(1366, 768)
        self.setAutoFillBackground(False) 

        self.ind=0

        namaPantai=[]
        npn = NamaPantai()  
        namaPan = npn.nama_pantai()
        for i in range(len(namaPan)):
            namaPantai.append(namaPan[i])
        self.label_pantai = namaPantai

        self.label_pantai_sort=self.getLabelSort()[0]
        self.label_pantai_sort2=self.getLabelSort()[1]
        self.label_pantai_sort3=self.getLabelSort()[2]
        self.label_pantai_sort4=self.getLabelSort()[3]
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1366, 768))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_Grafik = QtWidgets.QWidget()
        self.tab_Grafik.setObjectName("tab_Grafik")
        self.radioALL = QtWidgets.QRadioButton(self.tab_Grafik) 
        self.radioALL.setGeometry(QtCore.QRect(850, 30, 100, 21)) 
        self.radioDT = QtWidgets.QRadioButton(self.tab_Grafik) 
        self.radioDT.setGeometry(QtCore.QRect(950, 30, 100, 21)) 
        self.radioDT.setObjectName("radioDT")
        self.radioAK = QtWidgets.QRadioButton(self.tab_Grafik) 
        self.radioAK.setGeometry(QtCore.QRect(1050, 30, 100, 21)) 
        self.radioAK.setObjectName("radioAK")
        self.radioKB = QtWidgets.QRadioButton(self.tab_Grafik) 
        self.radioKB.setGeometry(QtCore.QRect(1150, 30, 100, 21)) 
        self.radioKB.setObjectName("radioKB")
        self.radioFS = QtWidgets.QRadioButton(self.tab_Grafik) 
        self.radioFS.setGeometry(QtCore.QRect(1250, 30, 100, 21)) 
        self.radioFS.setObjectName("radioFS")

        self.step = 0.8
        self.verticalScrollBar = QtWidgets.QScrollBar(
            self.tab_Grafik,
            sliderMoved=self.onAxisSliderMoved,
            pageStep=self.step * 10)
        self.verticalScrollBar.setGeometry(QtCore.QRect(30, 620, 1300, 20))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.verticalScrollBar.setObjectName("verticalScrollBar")

        self.comboBox = QtWidgets.QComboBox(self.tab_Grafik)
        self.comboBox.setGeometry(QtCore.QRect(110, 30, 161, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("=Pilih=")
        self.tmp_res = None
    
        for i in range(len(namaPan)):
            self.comboBox.addItem(namaPan[i])
        self.label_6 = QtWidgets.QLabel(self.tab_Grafik)
        self.label_6.setGeometry(QtCore.QRect(40, 30, 81, 16))
        self.label_6.setObjectName("label_6")
        self.graphicsView = QChartView(self.tab_Grafik)
        self.graphicsView.setGeometry(QtCore.QRect(30, 70, 1300, 520))
        self.graphicsView.setObjectName("graphicsView")
        self.tabWidget.addTab(self.tab_Grafik, "")

        self.tab_Klasifikasi = QtWidgets.QWidget()
        self.tab_Klasifikasi.setObjectName("tab_Klasifikasi")
        self.label = QtWidgets.QLabel(self.tab_Klasifikasi)
        self.label.setGeometry(QtCore.QRect(450, 70, 381, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_Klasifikasi)
        self.label_2.setGeometry(QtCore.QRect(170, 150, 131, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab_Klasifikasi)
        self.label_3.setGeometry(QtCore.QRect(170, 370, 200, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab_Klasifikasi)
        self.label_4.setGeometry(QtCore.QRect(850, 150, 71, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.tab_Klasifikasi)
        self.label_5.setGeometry(QtCore.QRect(850, 370, 71, 21))
        self.label_5.setObjectName("label_5")
        self.inputUlasan = QtWidgets.QPlainTextEdit(self.tab_Klasifikasi)
        self.inputUlasan.setGeometry(QtCore.QRect(170, 200, 500, 121))
        self.inputUlasan.setFont(QFont('Times',15))
        self.inputUlasan.setObjectName("inputUlasan")
        self.hasilPreprocessing = QtWidgets.QPlainTextEdit(self.tab_Klasifikasi)
        self.hasilPreprocessing.setGeometry(QtCore.QRect(170, 420, 500, 121))
        self.hasilPreprocessing.setFont(QFont('Times',15))
        self.hasilPreprocessing.setObjectName("hasilPreprocessing")
        self.hasilSentimen = QtWidgets.QPlainTextEdit(self.tab_Klasifikasi)
        self.hasilSentimen.setGeometry(QtCore.QRect(850, 200, 300, 121))
        self.hasilSentimen.setObjectName("hasilSentimen")
        self.hasilSentimen.setFont(QFont('Times',15))
        self.hasilKategori = QtWidgets.QPlainTextEdit(self.tab_Klasifikasi)
        self.hasilKategori.setGeometry(QtCore.QRect(850, 420, 300, 121))
        self.hasilKategori.setObjectName("hasilKategori")
        self.hasilKategori.setFont(QFont('Times',15))
        self.buttonPrediksi = QtWidgets.QPushButton(self.tab_Klasifikasi)
        self.buttonPrediksi.setGeometry(QtCore.QRect(170, 570, 111, 30))
        self.buttonPrediksi.setObjectName("buttonPrediksi")
        self.buttonDetail = QtWidgets.QPushButton(self.tab_Klasifikasi)
        self.buttonDetail.setGeometry(QtCore.QRect(350, 570, 111, 30))
        self.buttonDetail.setObjectName("buttonDetail")
        self.buttonClear = QtWidgets.QPushButton(self.tab_Klasifikasi)
        self.buttonClear.setGeometry(QtCore.QRect(1039, 570, 111, 30))
        self.buttonClear.setObjectName("buttonClear")
        self.tabWidget.addTab(self.tab_Klasifikasi, "")
        self.tab_Pengujian = QtWidgets.QWidget()
        self.tab_Pengujian.setObjectName("tab_Pengujian")
        self.graphicsView_sentimen = QChartView(self.tab_Pengujian)
        self.graphicsView_sentimen.setGeometry(QtCore.QRect(50, 100, 600, 560))
        self.graphicsView_sentimen.setObjectName("graphicsView")
        self.graphicsView_kategori = QChartView(self.tab_Pengujian)
        self.graphicsView_kategori.setGeometry(QtCore.QRect(700, 100, 600, 560))
        self.graphicsView_kategori.setObjectName("graphicsView")
        self.label_ujisentimen = QtWidgets.QLabel(self.tab_Pengujian)
        self.label_ujisentimen.setGeometry(QtCore.QRect(200, 50, 300, 31))
        self.label_ujisentimen.setObjectName("label_ujisentimen")
        self.label_ujiKategori = QtWidgets.QLabel(self.tab_Pengujian)
        self.label_ujiKategori.setGeometry(QtCore.QRect(850, 50, 300, 31))
        self.label_ujiKategori.setObjectName("label_ujiKategori")
        self.tabWidget.addTab(self.tab_Pengujian, "")
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.comboBox.activated.connect(self.selectionChange)
        self.ALL()
        self.buttonPrediksi.clicked.connect(self.on_click)
        self.buttonClear.clicked.connect(self.delete)
        self.buttonDetail.clicked.connect(self.openDetails)
        self.graphicsView_sentimen.setChart(self.grafik_uji_sentimen())
        self.graphicsView_kategori.setChart(self.grafik_uji_kategori())
        self.radioALL.setChecked(True)
        self.radioALL.toggled.connect(self.ALL)
        self.radioDT.toggled.connect(self.DT)
        self.radioAK.toggled.connect(self.AK)
        self.radioKB.toggled.connect(self.KB)
        self.radioFS.toggled.connect(self.FS)
            

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Analisis Sentimen"))
        self.buttonPrediksi.setText(_translate("MainWindow", "Prediksi"))
        self.buttonDetail.setText(_translate("MainWindow", "Detail"))
        self.buttonClear.setText(_translate("MainWindow", "Hapus"))
        self.radioALL.setText(_translate("MainWindow", "Kategori"))
        self.radioDT.setText(_translate("MainWindow", "Daya Tarik"))
        self.radioAK.setText(_translate("MainWindow", "Aksesbilitas"))
        self.radioKB.setText(_translate("MainWindow", "Kebersihan"))
        self.radioFS.setText(_translate("MainWindow", "Fasilitas"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; font-weight:600;\">KLASIFIKASI ULASAN GOOGLE MAPS</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600;\">Masukkan Ulasan :</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Hasil Preprocessing :</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600;\">Sentimen :</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600;\">Kategori :</span></p></body></html>"))
        self.label_ujisentimen.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Grafik Pengujian Sentimen</span></p></body></html>"))
        self.label_ujiKategori.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Grafik Pengujian Kategori</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Klasifikasi), _translate("MainWindow", "Klasifikasi"))
        self.comboBox.setItemText(0, _translate("MainWindow", "-Pilih-"))
        np = NamaPantai()  
        hasil = np.nama_pantai()
        for i in range(len(hasil)):
            self.comboBox.setItemText(i+1, _translate("MainWindow", hasil[i]))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt;\">Pilih Pantai :</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Grafik), _translate("MainWindow", "Grafik"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Pengujian), _translate("MainWindow", "Pengujian"))


    def selectionChange(self):
        pilih_pantai = str(self.comboBox.currentText())
        if pilih_pantai == '-Pilih-':
            self.ALL()
            return
        gf = Grafik(pilih_pantai)
        hasil = gf.hasil_grafik()
        hasil_dayatarik= hasil[0]
        hasil_aksesbilitas= hasil[1]
        hasil_kebersihan = hasil[2]
        hasil_fasilitas =hasil[3]
        total_pantai =hasil[4]
        daya_tarik=(hasil_dayatarik/total_pantai)*100
        aksesbilitas=(hasil_aksesbilitas/total_pantai)*100
        kebersihan=(hasil_kebersihan/total_pantai)*100
        fasilitas=(hasil_fasilitas/total_pantai)*100
        nilai = {'Daya Tarik': daya_tarik, 'Aksesbilitas': aksesbilitas , 'Kebersihan':kebersihan,'Fasilitas':fasilitas }
        maks=max(nilai['Daya Tarik'],nilai['Aksesbilitas'],nilai['Kebersihan'],nilai['Fasilitas'])
        for i in nilai:
            if maks==nilai[i]:
                print (i)
        set0 = QBarSet(pilih_pantai)
        set0 << daya_tarik << aksesbilitas << kebersihan << fasilitas 
        series = QBarSeries()
        series.append(set0)
        series.setLabelsVisible(True)
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle('Pantai '+ str(pilih_pantai))

        labelnya=["Daya Tarik","Aksesbilitas","Kebersihan","Fasilitas"]
 
        series = QPieSeries()
        series.append("dt",daya_tarik)
        series.append("ak",aksesbilitas)
        series.append("keb",kebersihan)
        series.append("fas",fasilitas)

        slice = QPieSlice()
        slice = series.slices()[0]
        slice.setExploded(False)
        slice.setLabelVisible(True)
        slice.setBrush(QtGui.QColor("blue"))

        slice = QPieSlice()
        slice = series.slices()[1]
        slice.setExploded(False)
        slice.setLabelVisible(True)
        slice.setBrush(QtGui.QColor("red"))

        slice = QPieSlice()
        slice = series.slices()[2]
        slice.setExploded(False)
        slice.setLabelVisible(True)
        slice.setBrush(QtGui.QColor("green"))

        slice = QPieSlice()
        slice = series.slices()[3]
        slice.setExploded(False)
        slice.setLabelVisible(True)
        slice.setBrush(QtGui.QColor("orange"))
        i=0
        for slice in series.slices():
            slice.setLabel(labelnya[i]+"  {:.1f}%".format(100 * slice.percentage()))
            i=i+1
 
        chart = QChart()
        chart.legend().hide()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.setAnimationOptions(QChart.SeriesAnimations) 
        self.graphicsView.setChart(chart)

    def ALL(self):
        self.ind=0
        set0 = QBarSet('Daya Tarik')
        set1 = QBarSet('Aksesbilitas')
        set2 = QBarSet('Kebersihan')
        set3 = QBarSet('Fasilitas')

        set0.setColor(QtGui.QColor("blue"))
        set1.setColor(QtGui.QColor("red"))
        set2.setColor(QtGui.QColor("green"))
        set3.setColor(QtGui.QColor("orange"))
        gf = Grafik_2()

        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        hasil = gf.hasil_dayatarik()
        for i in range(len(hasil)):
           set0.append((hasil[i]/jml_pantai[i])*100)

        hasil2 = gf.hasil_aksesbilitas()      
        for i in range(len(hasil2)):
           set1.append((hasil2[i]/jml_pantai[i])*100)

        hasil3 = gf.hasil_kebersihan()
        for i in range(len(hasil3)):
           set2.append((hasil3[i]/jml_pantai[i])*100)

        hasil4 = gf.hasil_fasilitas()
        for i in range(len(hasil4)):
           set3.append((hasil4[i]/jml_pantai[i])*100)
      
        series = QBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)
        series.append(set3)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsInsideEnd)
        series.setLabelsAngle(-90)

        self.chart = QChart()
        self.chart.addSeries(series)
        self.chart.setTitle('Grafik Prosentase Ulasan Pantai')
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        axisY = QValueAxis()
        axisY.setTitleText("Prosentase (%)")
        axisY.applyNiceNumbers()
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.createDefaultAxes()
        self.graphicsView.setChart(self.chart)
        self.adjust_axes(0, 2)
        self.lims = np.array([0, 6])
        self.onAxisSliderMoved(self.verticalScrollBar.value())

    def getLabelSort(self):
        gf = Grafik_2()
        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        value=[]
        value2=[]
        value3=[]
        value4=[]
        hasil = gf.hasil_dayatarik()
        for i in range(len(hasil)):
           value.append((hasil[i]/jml_pantai[i])*100)
        sorting=sorted(value,reverse=True)
        index=np.argsort(value)[::-1]
        label_sorting=[]
        for i in index :
          label_sorting.append(self.label_pantai[i])

        hasil2 = gf.hasil_aksesbilitas()
        for i in range(len(hasil2)):
           value2.append((hasil2[i]/jml_pantai[i])*100)
        sorting2=sorted(value2,reverse=True)
        index2=np.argsort(value2)[::-1]
        label_sorting2=[]
        for i in index2 :
          label_sorting2.append(self.label_pantai[i])

        hasil3 = gf.hasil_kebersihan()
        for i in range(len(hasil3)):
            value3.append((hasil3[i]/jml_pantai[i])*100)
        sorting3=sorted(value3,reverse=True)
        index3=np.argsort(value3)[::-1]
        label_sorting3=[]
        for i in index3 :
            label_sorting3.append(self.label_pantai[i])

        hasil4 = gf.hasil_fasilitas()
        for i in range(len(hasil4)):
            value4.append((hasil4[i]/jml_pantai[i])*100)
        sorting4=sorted(value4,reverse=True)
        index4=np.argsort(value4)[::-1]
        label_sorting4=[]
        for i in index4 :
            label_sorting4.append(self.label_pantai[i])
        return(label_sorting,label_sorting2,label_sorting3,label_sorting4)   


    def adjust_axes(self, value_min, value_max):
        if value_max > 35:
            return
        self.chart.createDefaultAxes()
        self.chart.axisX().setRange(
            str(value_min), str(value_max)
        )
        if self.ind == 0 :
            for i in range(value_min, value_max+1 if value_max < 36 else len(self.label_pantai)):
                self.chart.axisX().replace(str(i), self.label_pantai[i-1 if i > 0 else i])  
        if self.ind  == 1 :
            for i in range(value_min, value_max+1 if value_max < 36 else len(self.label_pantai_sort)):
                self.chart.axisX().replace(str(i), self.label_pantai_sort[i-1 if i > 0 else i])   
        if self.ind  == 2 :
            for i in range(value_min, value_max+1 if value_max < 36 else len(self.label_pantai_sort2)):
                self.chart.axisX().replace(str(i), self.label_pantai_sort2[i-1 if i > 0 else i])  
        if self.ind  == 3 :
            for i in range(value_min, value_max+1 if value_max < 36 else len(self.label_pantai_sort3)):
                self.chart.axisX().replace(str(i), self.label_pantai_sort3[i-1 if i > 0 else i])  
        if self.ind  == 4 :
            for i in range(value_min, value_max+1 if value_max < 36 else len(self.label_pantai_sort4)):
                self.chart.axisX().replace(str(i), self.label_pantai_sort4[i-1 if i > 0 else i])      


    @QtCore.pyqtSlot(int)
    def onAxisSliderMoved(self, value):
        r = value / ((1 + self.step) * 10)
        l1 = self.lims[0] + r * np.diff(self.lims)
        l2 = l1 + np.diff(self.lims) * self.step
        self.adjust_axes(math.floor(l1), math.ceil(l2))
            

    def DT(self):
        self.ind=1
        gf = Grafik_2()
        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        value=[]
        hasil = gf.hasil_dayatarik()
        for i in range(len(hasil)):
           value.append((hasil[i]/jml_pantai[i])*100)
        sorting=sorted(value,reverse=True)
        index=np.argsort(value)[::-1]
        label_sorting=[]
        for i in index :
          label_sorting.append(self.label_pantai[i])

        set0 = QBarSet('Daya Tarik')
        set0.setColor(QtGui.QColor("blue"))
        for i in range(len(hasil)):
           set0.append(sorting)
                
        series = QBarSeries()
        series.append(set0)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsInsideEnd)
        self.chart = QChart()
        self.chart.addSeries(series)
        self.chart.setTitle('Grafik Prosentase Ulasan Pantai')
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        axisX = QBarCategoryAxis()
        axisX.setLabelsAngle(-90)

        axisY = QValueAxis()
        axisY.setTitleText("Prosentase (%)")
        self.chart.addAxis(axisX, Qt.AlignBottom)
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)
        axisY.applyNiceNumbers()
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.createDefaultAxes()
        self.graphicsView.setChart(self.chart)
        self.lims = np.array([0, 10])
        self.onAxisSliderMoved(self.verticalScrollBar.value())


    def AK(self):
        self.ind=2
        gf = Grafik_2()
        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        value=[]
        hasil = gf.hasil_aksesbilitas()
        for i in range(len(hasil)):
           value.append((hasil[i]/jml_pantai[i])*100)
        sorting=sorted(value,reverse=True)

        set0 = QBarSet('Aksesbilitas')
        set0.setColor(QtGui.QColor("red"))  
        for i in range(len(hasil)):
           set0.append(sorting)   
        series = QBarSeries()
        series.append(set0)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsInsideEnd)
        
        self.chart = QChart()
        self.chart.addSeries(series)
        self.chart.setTitle('Grafik Prosentase Ulasan Pantai')
        self.chart.setAnimationOptions(QChart.SeriesAnimations)

        axisY = QValueAxis()
        axisY.setTitleText("Prosentase (%)")
        axisY.applyNiceNumbers()
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.createDefaultAxes()
        self.graphicsView.setChart(self.chart)
        self.adjust_axes(0, 2)
        self.lims = np.array([0, 10])
        self.onAxisSliderMoved(self.verticalScrollBar.value())

    def KB(self):
        self.ind=3
        gf = Grafik_2()
        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        value=[]
        hasil = gf.hasil_kebersihan()
        for i in range(len(hasil)):
           value.append((hasil[i]/jml_pantai[i])*100)
        sorting=sorted(value,reverse=True)

        set0 = QBarSet('Kebersihan')
        set0.setColor(QtGui.QColor("green"))  
        for i in range(len(hasil)):
           set0.append(sorting)   
 
        series = QBarSeries()
        series.append(set0)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsInsideEnd)
        
        self.chart = QChart()
        self.chart.addSeries(series)
        self.chart.setTitle('Grafik Prosentase Ulasan Pantai')
        self.chart.setAnimationOptions(QChart.SeriesAnimations)

        axisY = QValueAxis()
        axisY.setTitleText("Prosentase (%)")
        axisY.applyNiceNumbers()
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.createDefaultAxes()
        self.graphicsView.setChart(self.chart)
        self.adjust_axes(0, 2)
        self.lims = np.array([0, 10])
        self.onAxisSliderMoved(self.verticalScrollBar.value())

    def FS(self):
        self.ind=4
        gf = Grafik_2()
        jml=gf.jumlah_per_pantai()
        jml_pantai=[]
        for i in range(len(jml)):
            jml_pantai.append(jml[i])

        value=[]
        hasil = gf.hasil_fasilitas()
        for i in range(len(hasil)):
           value.append((hasil[i]/jml_pantai[i])*100)
        sorting=sorted(value,reverse=True)

        set0 = QBarSet('Fasilitas')
        set0.setColor(QtGui.QColor("orange"))  
        for i in range(len(hasil)):
           set0.append(sorting)   
      
        series = QBarSeries()
        series.append(set0)
        series.setLabelsVisible(True)
        series.setLabelsPosition(QAbstractBarSeries.LabelsInsideEnd)
                
        self.chart = QChart()
        self.chart.addSeries(series)
        self.chart.setTitle('Grafik Prosentase Ulasan Pantai')
        self.chart.setAnimationOptions(QChart.SeriesAnimations)

        axisY = QValueAxis()
        axisY.setTitleText("Prosentase (%)")
        axisY.applyNiceNumbers()
        self.chart.addAxis(axisY, Qt.AlignLeft)
        series.attachAxis(axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        self.chart.createDefaultAxes()
        self.graphicsView.setChart(self.chart)
        self.adjust_axes(0, 2)
        self.lims = np.array([0, 10])
        self.onAxisSliderMoved(self.verticalScrollBar.value())


    def on_click(self):
        textboxValue = self.inputUlasan.toPlainText()
        cf = Classification(textboxValue)
        hasil = cf.klasifikasi()
        hasil_sentimen= hasil[0]
        hasil_kategori= hasil[1]
        token = hasil[2]
        predict_sentimen_positif=hasil[3]
        predict_sentimen_negatif=hasil[4]

        predict_DayaTarik=hasil[5]
        predict_Aksesbilitas=hasil[6]
        predict_Kebersihan=hasil[7]
        predict_Fasilitas=hasil[8]
        if self.inputUlasan.toPlainText()!='' :
            self.inputUlasan.setPlainText(textboxValue)
            self.hasilPreprocessing.setPlainText(token)
            self.hasilSentimen.setPlainText(hasil_sentimen)
            self.hasilKategori.setPlainText(hasil_kategori)
            self.tmp_res = (predict_sentimen_positif,predict_sentimen_negatif,predict_DayaTarik,predict_Aksesbilitas,predict_Kebersihan,predict_Fasilitas)

        return self.tmp_res


    def delete(self):
        self.inputUlasan.setPlainText("")
        self.hasilPreprocessing.setPlainText("")
        self.hasilSentimen.setPlainText("")
        self.hasilKategori.setPlainText("")


    def grafik_uji_sentimen(self):
        with open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Pengujian/hasil_uji_sentimen_kfold.json') as f:
            dataa=json.load(f)
        print (dataa)
        acc= dataa['acc']
        pres= dataa['presisi']
        rec = dataa['recall']
        set0 = QBarSet('Akurasi')
        set1 = QBarSet('Presisi')
        set2 = QBarSet('Recall')

        for i in range(len(acc)):
           set0.append(acc[i]*100)

        for i in range(len(pres)):
           set1.append(pres[i]*100)

        for i in range(len(rec)):
           set2.append(rec[i]*100)

        series = QBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)

        set0.setColor(QtGui.QColor("navy"))
        set1.setColor(QtGui.QColor("yellow"))
        set2.setColor(QtGui.QColor("red"))

        chart = QChart()
        chart.addSeries(series)
        chart.setAnimationOptions(QChart.SeriesAnimations)
        label_pantai = ['K=1','K=2', 'K=3','K=4','K=5']
        axisX = QBarCategoryAxis()
        axisX.append(label_pantai)
        axisX.setLabelsAngle(0)
        axisY = QValueAxis()
        axisX.setTitleText("K-Fold Cross Validation")
        axisY.setTitleText("Prosentase (%)")
        axisY.setRange(0, max(set0))
        axisY.setMinorTickCount(5)
        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        return(chart)

    def grafik_uji_kategori(self):
        with open('C:/Users/asus/Desktop/SKRIPSWEET BISMILLAH/MODUL PROGRAM/Modul Program Bismillah/Pengujian/hasil_uji_kategori_kfold.json') as f:
            dataa=json.load(f)
        print (dataa)
        acc= dataa['acc']
        pres= dataa['presisi']
        rec = dataa['recall']
        set0 = QBarSet('Akurasi')
        set1 = QBarSet('Presisi')
        set2 = QBarSet('Recall')

        for i in range(len(acc)):
           set0.append(acc[i]*100)

        for i in range(len(pres)):
           set1.append(pres[i]*100)

        for i in range(len(rec)):
           set2.append(rec[i]*100)

        series = QBarSeries()
        series.append(set0)
        series.append(set1)
        series.append(set2)
        set0.setColor(QtGui.QColor("navy"))
        set1.setColor(QtGui.QColor("yellow"))
        set2.setColor(QtGui.QColor("red"))

        chart = QChart()
        chart.addSeries(series)
        chart.setAnimationOptions(QChart.SeriesAnimations)
        label_pantai = ['K=1','K=2', 'K=3','K=4','K=5']
        axisX = QBarCategoryAxis()
        axisX.append(label_pantai)
        axisX.setLabelsAngle(0)
        axisY = QValueAxis()
        axisX.setTitleText("K-Fold Cross Validation")
        axisY.setTitleText("Prosentase (%)")
        axisY.setRange(0, max(set0))
        axisY.setMinorTickCount(5)

        chart.addAxis(axisX, Qt.AlignBottom)
        chart.addAxis(axisY, Qt.AlignLeft)
        chart.legend().setVisible(True)
        chart.legend().setAlignment(Qt.AlignBottom)
        return(chart)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Ui_MainWindow()
    w.show()
    sys.exit(app.exec_())