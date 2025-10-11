import nltk
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download('stopwords')

class BilingualStemmer:
    
    def __init__(self, additional_language=None):
        self.languages = ['english']
        self.stop_words = set()
        self.stemmers = {}
        
        # Always initialize English
        try:
            english_stop_words = set(stopwords.words('english'))
            self.stop_words.update(english_stop_words)
            self.stemmers['english'] = SnowballStemmer('english')
            log.info("Initialized stemmer for language: english")
        except OSError:
            log.warning("English stopwords not found")
        
        # Initialize additional language if provided
        if additional_language:
            self.languages.append(additional_language)
            try:
                additional_stop_words = set(stopwords.words(additional_language))
                self.stop_words.update(additional_stop_words)
            except OSError:
                log.warning(f"Stopwords for language '{additional_language}' not found")
            
            if additional_language in SnowballStemmer.languages:
                self.stemmers[additional_language] = SnowballStemmer(additional_language)
                log.info(f"Initialized stemmer for language: {additional_language}")
            else:
                log.warning(f"Stemmer for language '{additional_language}' not available")
    
    def _detect_language(self, word):        
        if any('\u0400' <= char <= '\u04FF' for char in word):
            return self.languages[1] if len(self.languages) > 1 else 'english'
        return 'english'
    
    def preprocess(self, text):        
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        stemmed_tokens = []
        for word in tokens:
            lang = self._detect_language(word)
            if lang in self.stemmers:
                stemmed_word = self.stemmers[lang].stem(word)
                stemmed_tokens.append(stemmed_word)
            else:
                stemmed_tokens.append(word)
                
        return stemmed_tokens

    def __call__(self, text):
        return self.preprocess(text)