from tkinter import *
from tkinter import filedialog
import docx2txt
import dokuman_modulu


def dokuman_yukle():
    dosya_yolu = filedialog.askopenfilename(initialdir="/", title="Doküman Seç", filetypes=(("Word Dosyaları", "*.docx"), ("Tüm Dosyalar", "*.*")))
    if dosya_yolu:
        dokuman_icerik = docx2txt.process(dosya_yolu)
        dokuman_modulu.dokuman_icerik = dokuman_icerik



# Masaüstü uygulaması penceresi oluşturma
pencere = Tk()
pencere.title("Doküman Yükleme Uygulaması")

# Etiket oluşturma
etiket = Label(pencere, text="Bir Word dokümanı seçin ve yükleyin.")
etiket.pack()

# Düğme oluşturma
dugme = Button(pencere, text="Doküman Yükle", command=dokuman_yukle)
dugme.pack()


# Uygulamayı çalıştırma
pencere.mainloop()