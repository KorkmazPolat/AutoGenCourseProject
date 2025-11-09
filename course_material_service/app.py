from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from datetime import datetime
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# Flask, FastAPI'deki gibi kolayca statik klasörler yaratmaz.
# 'exported' klasöründeki statik kurs dosyalarını sunmak için bir rota ekliyoruz.
@app.route('/exported/<path:filename>')
def serve_exported_files(filename):
    return send_from_directory('exports', filename)

# Flask'ta form gönderme, dosya okuma ve exports listeleme rotaları ekleniyor.

@app.route("/", methods=['GET', 'POST'])
def index():
    # Form gönderiminden sonra sayfayı yeniden yüklerken eski değerleri tutmak için
    values = {} 
    
    # Varsayılan olarak index.html'i render et.
    return render_template("index.html", current_year=datetime.now().year, values=values)

@app.route("/exports")
def list_exports():
    """Kayıtlı kursları listeler ve yeni tasarımlı exports.html şablonunu kullanır."""
    
    exports_dir = Path("exports")
    exports_list = []
    
    if exports_dir.is_dir():
        for item in exports_dir.iterdir():
            if item.is_dir():
                folder_name = item.name 
                index_path = item / "index.html"
                
                if index_path.exists():
                    # Flask'ın statik dosya sunumu için doğru URL yolu:
                    url = f"/exported/{folder_name}/index.html" 
                    
                    # Kurs başlığını klasör adından daha okunaklı hale getir
                    # (Tarih ve saat kısmını keser)
                    display_name = folder_name.replace('-', ' ').split(' 202')[0]
                    
                    exports_list.append((url, display_name.title()))
    
    # Yeni exports.html şablonunu Dark Mode tasarımıyla render et
    return render_template("exports.html", exports=exports_list)


@app.route("/create-full-course", methods=['POST'])
def create_full_course():
    """Form verilerini alır, kurs oluşturma mantığını çalıştırır ve sonucu döndürür."""
    
    # 1. Form verilerini al
    course_title = request.form.get('course_title')
    audience = request.form.get('audience')
    tone = request.form.get('tone')
    duration_minutes = request.form.get('duration_minutes')
    learning_outcomes_raw = request.form.get('learning_outcomes')
    
    # Basit doğrulama (daha karmaşık doğrulama front-end'de yapılmalı)
    if not course_title or not learning_outcomes_raw:
        # Hata durumunda formu eski verilerle yeniden göster
        values = request.form
        return render_template("index.html", current_year=datetime.now().year, values=values, error="Please fill in required fields.")


    
    modules_output = [...] # Bu, AI'dan gelen veridir
    blueprint = {} # AI'dan gelen kurs planı
    

    
    return render_template('full_course.html', 
                           course_title=course_title, 
                           learning_outcomes=learning_outcomes_raw.split('\n'),
                           blueprint=blueprint, 
                           modules_output=modules_output, 
                           voice="en_us_male_standard") # Bu veriler AI'dan gelmelidir.


if __name__ == "__main__":
    from flask import send_from_directory # send_from_directory import edilmeli

    # Flask'ta port 8000'i kullanmak için:
    app.run(debug=True, port=8000)