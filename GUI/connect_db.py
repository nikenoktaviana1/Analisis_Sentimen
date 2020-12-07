import matplotlib.pyplot as plt
import pymysql

def jenis_kategori(p, k):
	db = pymysql.connect(
    host='localhost',
    user='root',
    passwd='',
    port=3306,
    db='database_analisis_sentimen'
    )

	pantai=p
	kategori=k
	tot_positif=0
	cursor = db.cursor()
	sql= "SELECT * from hasil_klasifikasi_2 JOIN daftar_pantai ON hasil_klasifikasi_2.id_pantai=daftar_pantai.id WHERE hasil_klasifikasi_2.klasifikasi_sentimen='Positif' and daftar_pantai.nama_pantai='%s' and hasil_klasifikasi_2.klasifikasi_kategori= '%s'"%(pantai,kategori)
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		for row in results:
			tot_positif=tot_positif+1

	except :
		print("error")
	return(tot_positif)

def total_pantai(p):
	db = pymysql.connect(
    host='localhost',
    user='root',
    passwd='',
    port=3306,
    db='database_analisis_sentimen'
    )

	pantai=p
	tot_pantai=0
	cursor = db.cursor()
	sql="SELECT * from hasil_klasifikasi_2 JOIN daftar_pantai ON hasil_klasifikasi_2.id_pantai=daftar_pantai.id WHERE daftar_pantai.nama_pantai='%s'"%(pantai)
	# sql="SELECT * from hasil_klasifikasi where nama_pantai = '%s' "%(pantai)
	try:
		cursor.execute(sql)
		results = cursor.fetchall()
		for row in results:
			tot_pantai=tot_pantai+1

	except :
		print("error")
	return(tot_pantai)


class Grafik:
	def __init__(self,pilih_pantai):
		self.pilih_pantai = pilih_pantai
	def hasil_grafik(self):
		pilih_pantai = self.pilih_pantai
		print(pilih_pantai)
		hasil_dayatarik=jenis_kategori(pilih_pantai,"Daya Tarik")
		hasil_aksesbilitas=jenis_kategori(pilih_pantai,"Aksesbilitas")
		hasil_kebersihan=jenis_kategori(pilih_pantai,"Kebersihan")
		hasil_fasilitas=jenis_kategori(pilih_pantai,"Fasilitas")
		jumlah_ulasan=total_pantai(pilih_pantai)
		return(hasil_dayatarik,hasil_aksesbilitas,hasil_kebersihan,hasil_fasilitas,jumlah_ulasan)
		db.close()

class Grafik_2:
	def hasil_dayatarik(self):
		np = NamaPantai()
		hasil_pantai=[]
		hasil = np.nama_pantai()
		for i in range(len(hasil)):
			hasil_pantai.append(jenis_kategori(hasil[i],'Daya Tarik'))
		return(hasil_pantai)

	def hasil_aksesbilitas(self):
		np = NamaPantai()
		hasil_pantai=[]
		hasil = np.nama_pantai()
		for i in range(len(hasil)):
			hasil_pantai.append(jenis_kategori(hasil[i],'Aksesbilitas'))
		return(hasil_pantai)

	def hasil_kebersihan(self):
		np = NamaPantai()
		hasil_pantai=[]
		hasil = np.nama_pantai()
		for i in range(len(hasil)):
			hasil_pantai.append(jenis_kategori(hasil[i],'Kebersihan'))
		return(hasil_pantai)

	def hasil_fasilitas(self):
		np = NamaPantai()
		hasil_pantai=[]
		hasil = np.nama_pantai()
		for i in range(len(hasil)):
			hasil_pantai.append(jenis_kategori(hasil[i],'Fasilitas'))
		return(hasil_pantai)
	
	def jumlah_per_pantai(self):
		np = NamaPantai()
		hasil_pantai=[]
		hasil = np.nama_pantai()
		for i in range(len(hasil)):
			hasil_pantai.append(total_pantai(hasil[i]))
		return(hasil_pantai)
		db.close()

class NamaPantai:
	def nama_pantai(self):
		db = pymysql.connect(
			host='localhost',
			user='root',
			passwd='',
			port=3306,
			db='database_analisis_sentimen')
		cursor = db.cursor()
		sql="SELECT * from daftar_pantai "
		try:
			cursor.execute(sql)
			nama_pantai=[]
			results = cursor.fetchall()
			for row in results:
				nama_pantai.append(row[1])
		except :
			print("error")
		return(nama_pantai)



