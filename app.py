from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

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
    'memuaskan': 'puas', 'oke': 'bagus', 'kece': 'bagus', 'nangis': 'sedih'
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
    'tidak_bagus', 'tidak_puas', 'kurang_bagus', 'tidak_keren'
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

if __name__ == '__main__':
    app.run(debug=True)

    