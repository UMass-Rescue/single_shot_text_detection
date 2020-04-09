import cv2
import numpy as np
from gingerit.gingerit import GingerIt
import pytesseract
from collections import Counter
from nltk import everygrams
from nltk.corpus import stopwords as sw
import en_core_web_sm
from spacy_langdetect import LanguageDetector
import re
import json
import os

def fetch_text_from_image_base(im, alphanumeric, correction=True):
    # Applying Image based correction..
    if correction:
        pxmin = np.min(im)
        pxmax = np.max(im)
        imgContrast = (im - pxmin) / (pxmax - pxmin) * 255
        kernel = np.ones((2, 2), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations = 2)
        imgMorph = imgMorph.astype(np.uint8)
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, threshold = cv2.threshold(im, 127, 255,cv2.THRESH_BINARY)
    else:
        threshold = im
    text = pytesseract.image_to_string(threshold)
    if alphanumeric:
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

def make_text_corrections(txt, custom_stopwords, top_k_ngrams, upper_ngram, known_list_corrections):
    # Module Initialization
    parser = GingerIt()
    nlp = en_core_web_sm.load()
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    if len(known_list_corrections) == 0:
        known_list_corrections = {'vou': 'you', 'ves':'yes', 'vea': 'yea'}

    corrected_txt = []
    for word in txt.split(' '):
        flag = False
        for key in known_list_corrections.keys():
            if key in word:
                word = word.replace(key, known_list_corrections.get(key))
                corrected_txt.append(word)
                flag = True
        if not flag and word != '':
            corrected_txt.append(word)
    coerced_word = ' '.join(corrected_txt)

    # Make grammatical_corrections
    all_changes = [(change.get('start'), len(change.get('text')), change.get('correct')) 
                   for change in parser.parse(coerced_word).get('corrections')]
    all_changes.sort(key=lambda item: item[0])
    grammatically_corrected_str = ''
    current_ptr = 0
    for start_index, length, suggested in all_changes:
        grammatically_corrected_str += coerced_word[current_ptr:start_index] + ' %s ' % suggested
        current_ptr = start_index + length
        
    # Stopword removal
    stopwords = list(set(sw.words('english') + custom_stopwords))
    filtered_final_str = [word for word in grammatically_corrected_str.split() 
                          if word.lower() not in stopwords and len(word) > 1]
    filtered_final_str = ' '.join(filtered_final_str)
    
    # Creating N-grams
    bi_trigrams = dict(Counter(list(everygrams(filtered_final_str.lower().split(), 2, upper_ngram))))
    bi_trigrams_tup = [(bi_trigrams.get(key), key) for key in bi_trigrams.keys()]
    bi_trigrams_tup.sort(key=lambda item: item[0], reverse=True)
    
    # Language detection
    import os
    lang_mapper_file = '../dataset/metadata/spacy_lang_mapper.json'
    if os.path.isfile(lang_mapper_file) is not True:
        spacy_lang_mapper = spacy_lang_mapper_dict()
    else:
        spacy_lang_mapper = json.loads(open(lang_mapper_file).read())
    detect_language = spacy_lang_mapper.get(nlp(filtered_final_str)._.language['language'])
    
    return (filtered_final_str, bi_trigrams_tup[:top_k_ngrams], detect_language)

def fetch_text_from_image(im, alphanumeric=True, perform_nlp=True, custom_stopwords=[], top_k_ngrams=10, upper_ngram=3, 
                         known_list_corrections=[], correction=True):
    if type(im) != np.ndarray:
        im = imread(im)
    original_text = fetch_text_from_image_base(im, alphanumeric, correction)
    if not perform_nlp:
        return original_text, None
    else:
        custom_stopwords = list(map(lambda x: x.lower(), custom_stopwords))
        corrected_text, top_ngrams, language = make_text_corrections(original_text, custom_stopwords, 
                                                                     top_k_ngrams, upper_ngram, known_list_corrections)
        return original_text, (corrected_text, top_ngrams, language)
    
def spacy_lang_mapper_dict():
    return {"de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fr": "French", "it": "Italian", "lt": "Lithuanian", "nb": "Norwegian Bokm\u00e5l", "nl": "Dutch", "pt": "Portuguese", "xx": "Multi-language", "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "da": "Danish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "ga": "Irish", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic", "ja": "Japanese", "kn": "Kannada", "ko": "Korean", "lb": "Luxembourgish", "lv": "Latvian", "mr": "Marathi", "pl": "Polish", "ro": "Romanian", "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "yo": "Yoruba", "zh": "Chinese"}