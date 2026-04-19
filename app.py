from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re
import string
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from googleapiclient.discovery import build

app = Flask(__name__)
YOUTUBE_API_KEY = 'AIzaSyAMMIUL6xcItmA8aIhIKNlRguUx9LJKIHY'

print("[INFO] Memuat Sistem dan Model NLP...")

# =====================================================================
# 1. LOAD SASTRAWI & STOPWORDS
# =====================================================================
stemmer = StemmerFactory().create_stemmer()
factory_stop = StopWordRemoverFactory()
stopwords_list = factory_stop.get_stop_words()

kata_penting = ['tapi', 'namun', 'cuma', 'hanya', 'sayang', 'padahal', 'kok', 'tidak', 'nggak', 'bukan', 'kurang', 'jangan', 'belum']
for word in kata_penting:
    if word in stopwords_list: stopwords_list.remove(word)

# =====================================================================
# 2. LOAD KAMUS TYPO & SARKASME
# =====================================================================
slang_dict = {
    'burik': 'jelek', 'busuk': 'jelek', 'bapuk': 'jelek', 'kentang': 'jelek', 
    'selangit': 'mahal', 'nguras': 'mahal', 'awet': 'bagus', 'gacor': 'bagus', 
    'mantul': 'bagus', 'keren': 'bagus', 'lemot': 'lambat', 'cepet': 'cepat', 
    'abis': 'habis', 'batre': 'baterai', 'casan': 'charger', 
    'bgaus': 'bagus', 'bgus': 'bagus', 'bgs': 'bagus', 'baguss': 'bagus',
    'jlk': 'jelek', 'jele': 'jelek', 'jelekk': 'jelek',
    'mhal': 'mahal', 'mahaal': 'mahal', 'murh': 'murah', 'muraah': 'murah',
    'cpt': 'cepat', 'lmot': 'lambat', 'lambt': 'lambat',
    'krn': 'karena', 'yg': 'yang', 'dgn': 'dengan', 'jernih': 'bagus', 'jelas': 'bagus',
    'mantap': 'bagus', 'jos': 'bagus', 'joss': 'bagus', 'top': 'bagus', 
    'memuaskan': 'puas', 'oke': 'bagus', 'kece': 'bagus', 'nangis': 'sedih',
    'murahan': 'jelek', 'kemurahan': 'murah', 'harga murah': 'murah',
    'kelas bawah': 'jelek', 'kelas rendah': 'jelek', 'hp murah': 'jelek',
    'terkesan murah': 'jelek', 'kayak murah': 'jelek'
}

sarkas_dict = {}
try:
    df_kamus = pd.read_csv('data/kamus_alay.csv')
    slang_dict.update(dict(zip(df_kamus.iloc[:, 0], df_kamus.iloc[:, 1])))

    # ✅ TAMBAHAN BARU: Load kamus gadget
    df_gadget = pd.read_csv('data/kamus_gadget.csv')
    kamus_gadget = dict(zip(df_gadget.iloc[:, 0], df_gadget.iloc[:, 1]))
    kamus_gadget.update(slang_dict)  # kamus_alay tetap prioritas jika bentrok
    slang_dict = kamus_gadget

except Exception as e:
    print(f"[WARNING] Kamus gagal dimuat: {e}")

# =====================================================================
# 3. LOAD MODEL SVM
# =====================================================================
try:
    vectorizer = joblib.load('models/vectorizer.pkl')
    svm_sentimen = joblib.load('models/model_sentimen_svm.pkl')
    svm_aspek = joblib.load('models/model_aspek_svm.pkl')
    model_ready = True
except Exception as e:
    model_ready = False
    print(f"[ERROR] Model gagal dimuat: {e}")

# =====================================================================
# 4. FUNGSI PREPROCESSING TEKS
# =====================================================================
def clean_text(text):
    text = str(text).lower()

    # ✅ TAMBAHAN: normalisasi frasa kontekstual SEBELUM proses lainnya
    frasa_negatif_konteks = [
        ('kayak hp murah', 'jelek'),
        ('kayak murah', 'jelek'),
        ('terkesan murah', 'jelek'),
        ('seperti murah', 'jelek'),
        ('kelas bawah', 'jelek'),
        ('kelas rendah', 'jelek'),
        ('spek rendah', 'jelek'),
        ('spesifikasi murahan', 'jelek'),
        ('desain murahan', 'jelek'),
        ('tidak premium', 'jelek'),
        ('kurang premium', 'jelek'),
    ]
    for frasa, ganti in frasa_negatif_konteks:
        text = text.replace(frasa, ganti)
    
    # 1. Sarkasme
    for frasa_asli, frasa_ganti in sarkas_dict.items():
        text = text.replace(str(frasa_asli), str(frasa_ganti))
    
    # 2. Hapus angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Normalisasi karakter berulang
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    
    # 4. Normalisasi slang
    words = text.split()
    text = ' '.join([slang_dict.get(w, w) for w in words])
    
    # 5. Baru ikat negasi
    text = re.sub(r'\b(tidak|kurang|belum|jangan|nggak|bukan)\s+(\w+)', r'\1_\2', text)
    
    # 6. Hapus stopwords
    words = text.split()
    text = ' '.join([w for w in words if w not in stopwords_list])
    
    # 7. Stemming
    return stemmer.stem(text)

def split_into_clauses(text):
    text = str(text).lower()
    
    aspek_keywords = ['kamera', 'harga', 'baterai', 'batre', 'desain']
    
    words = text.split()
    new_words = []
    
    for i, word in enumerate(words):
        if any(word.startswith(kw) for kw in aspek_keywords) and i > 0:
            if new_words[-1] not in ['dan', 'tapi', 'tetapi', 'sedangkan', 'padahal', 'namun'] and not any(p in new_words[-1] for p in [',', '.', ';']):
                new_words[-1] += ','
        new_words.append(word)
        
    text_with_shadow_commas = ' '.join(new_words)
    clauses = re.split(r',|\.|;| namun | tapi | tetapi | dan | sedangkan | padahal ', text_with_shadow_commas)
    return [c.strip() for c in clauses if len(c.strip().split()) >= 2]

def detect_and_split_reviews(text):
    """
    Mendeteksi apakah teks berisi 1 atau beberapa ulasan terpisah.
    Mengembalikan list of string, masing-masing adalah 1 ulasan.
    """
    text = text.strip()

    # Pola 1: Penomoran eksplisit → "1. ...", "1) ...", "(1) ..."
    pola_nomor = re.split(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|\(\d+\)\s*)', text)
    pola_nomor = [s.strip() for s in pola_nomor if s.strip()]
    if len(pola_nomor) >= 2:
        return pola_nomor, 'multi'

    # Pola 2: Baris baru ganda atau separator eksplisit
    pola_baris = re.split(r'\n{2,}|(?<=[.!?])\n(?=[A-Za-z0-9])', text)
    pola_baris = [s.strip() for s in pola_baris if len(s.strip()) > 10]
    if len(pola_baris) >= 2:
        return pola_baris, 'multi'

    # Pola 3: Separator manual → "||", " / ", " - " di awal
    pola_sep = re.split(r'\s*\|\|\s*', text)
    pola_sep = [s.strip() for s in pola_sep if s.strip()]
    if len(pola_sep) >= 2:
        return pola_sep, 'multi'

    # Default: 1 ulasan
    return [text], 'single'

# =====================================================================
# LEXICON FALLBACK
# =====================================================================
lexicon_positif = {
    'bagus', 'baik', 'keren', 'mantap', 'canggih', 'cepat', 'jernih',
    'tajam', 'awet', 'tahan', 'murah', 'terjangkau', 'puas', 'suka',
    'senang', 'recommended', 'oke', 'mantul', 'gacor', 'jos', 'worth',
    'memuaskan', 'responsif', 'smooth', 'lancar', 'kencang', 'hemat'
}
lexicon_negatif = {
    'jelek', 'buruk', 'lambat', 'lemot', 'mahal', 'rusak', 'hancur',
    'kecewa', 'zonk', 'sampah', 'payah', 'parah', 'butut', 'ampas',
    'boros', 'panas', 'overheat', 'retak', 'baret', 'lag', 'ngelag',
    'ancur', 'murahan', 'abal', 'kw', 'norak', 'bocor', 'jeblok',
    'tidak_bagus', 'tidak_puas', 'kurang_bagus', 'tidak_keren',
    'murahan', 'murahaan', 'kelas bawah', 'kelas rendah',
    'terkesan murah', 'kayak murah', 'seperti murah',
    'plastikan', 'ringkih', 'rapuh', 'goyang', 'kopong',
    'tidak premium', 'kurang premium', 'tidak elegan',
    'norak', 'kampungan', 'jadul', 'kuno', 'ketinggalan',
    'spesifikasi rendah', 'spek rendah', 'spek jelek',
}

def lexicon_score(text_bersih):
    words = text_bersih.split()
    return sum(1 for w in words if w in lexicon_positif) \
         - sum(1 for w in words if w in lexicon_negatif)

# =====================================================================
# 5. ROUTING WEB FLASK
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({'error': 'Model belum siap. Cek folder models/'}), 500

    data = request.get_json()
    komentar_asli = data.get('komentar', '')

    if not komentar_asli.strip():
        return jsonify({'results': [], 'blackbox': {}})

    # Deteksi dulu apakah 1 atau multi ulasan
    reviews, mode = detect_and_split_reviews(komentar_asli)

    semua_hasil = []

    for review_idx, review_text in enumerate(reviews):
        clauses = split_into_clauses(review_text)
        hasil_per_review = []

        for clause in clauses:
            cleaned_clause = clean_text(clause)
            if not cleaned_clause.strip():
                continue

            vec_text = vectorizer.transform([cleaned_clause])
            pred_aspek = svm_aspek.predict(vec_text)[0]

            try:
                prob_sentimen = svm_sentimen.predict_proba(vec_text)[0]
                max_prob = max(prob_sentimen)
                pred_sentimen = svm_sentimen.classes_[prob_sentimen.argmax()]

                THRESHOLD = 0.60
                if max_prob < THRESHOLD:
                    skor = lexicon_score(cleaned_clause)
                    if skor > 0:
                        pred_sentimen = 'Positif'
                    elif skor < 0:
                        pred_sentimen = 'Negatif'
                    else:
                        pred_sentimen = 'Netral'

                confidence_score = round(max_prob * 100, 2)
            except AttributeError:
                pred_sentimen = svm_sentimen.predict(vec_text)[0]
                confidence_score = 100.0

            hasil_per_review.append({
                'klausa': clause.capitalize(),
                'aspek': pred_aspek,
                'sentimen': pred_sentimen,
                'confidence': confidence_score,
                'teks_bersih': cleaned_clause
            })

        # ✅ BARU: Bungkus dengan info ulasan ke-N jika multi
        semua_hasil.append({
            'review_index': review_idx + 1,
            'review_text': review_text,
            'mode': mode,
            'clauses': hasil_per_review
        })

    return jsonify({
        'results': semua_hasil,
        'mode': mode,
        'blackbox': {
            'teks_asli': komentar_asli,
            'mode_deteksi': mode,
            'jumlah_ulasan': len(reviews),
            'hasil_split': [r['review_text'] for r in semua_hasil]
        }
    })

# =====================================================================
# FUNGSI YOUTUBE
# =====================================================================
def get_video_id(url):
    """Ekstrak video ID dari berbagai format URL YouTube."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_comments(video_id, max_comments=100):
    """Ambil komentar dari YouTube API, maksimal max_comments komentar."""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments
    except Exception as e:
        return {'error': str(e)}


@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    if not model_ready:
        return jsonify({'error': 'Model belum siap.'}), 500

    data = request.get_json()
    youtube_url = data.get('url', '')
    max_comments = int(data.get('max_comments', 50))

    # Ekstrak video ID
    video_id = get_video_id(youtube_url)
    if not video_id:
        return jsonify({'error': 'URL YouTube tidak valid.'}), 400

    # Ambil komentar
    comments = get_youtube_comments(video_id, max_comments)
    if isinstance(comments, dict) and 'error' in comments:
        return jsonify({'error': f"Gagal mengambil komentar: {comments['error']}"}), 500

    if not comments:
        return jsonify({'error': 'Tidak ada komentar ditemukan.'}), 404

    # Analisis tiap komentar
    hasil_semua = []
    ringkasan_sentimen = {'Positif': 0, 'Negatif': 0, 'Netral': 0, 'Ambigu': 0}
    ringkasan_aspek = {'Kamera': 0, 'Baterai': 0, 'Harga': 0, 'Desain': 0, 'Lainnya': 0}

    for komentar in comments:
        clauses = split_into_clauses(komentar)
        hasil_komentar = []

        for clause in clauses:
            cleaned = clean_text(clause)
            if not cleaned.strip():
                continue

            vec_text = vectorizer.transform([cleaned])
            pred_aspek = svm_aspek.predict(vec_text)[0]

            try:
                prob = svm_sentimen.predict_proba(vec_text)[0]
                max_prob = max(prob)
                pred_sentimen = svm_sentimen.classes_[prob.argmax()]

                if max_prob < 0.60:
                    skor = lexicon_score(cleaned)
                    if skor > 0: pred_sentimen = 'Positif'
                    elif skor < 0: pred_sentimen = 'Negatif'
                    else: pred_sentimen = 'Netral'

                confidence = round(max_prob * 100, 2)
            except AttributeError:
                pred_sentimen = svm_sentimen.predict(vec_text)[0]
                confidence = 100.0

            # Update ringkasan
            ringkasan_sentimen[pred_sentimen] = ringkasan_sentimen.get(pred_sentimen, 0) + 1
            if pred_aspek in ringkasan_aspek:
                ringkasan_aspek[pred_aspek] += 1
            else:
                ringkasan_aspek['Lainnya'] += 1

            hasil_komentar.append({
                'klausa': clause.capitalize(),
                'aspek': pred_aspek,
                'sentimen': pred_sentimen,
                'confidence': confidence
            })

        if hasil_komentar:
            hasil_semua.append({
                'komentar_asli': komentar,
                'hasil': hasil_komentar
            })

    total_klausa = sum(len(k['hasil']) for k in hasil_semua)

    return jsonify({
        'total_komentar': len(hasil_semua),
        'total_klausa': total_klausa,
        'ringkasan_sentimen': ringkasan_sentimen,
        'ringkasan_aspek': ringkasan_aspek,
        'detail': hasil_semua
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
