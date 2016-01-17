# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals
import unittest
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import os


class SnowballTest(unittest.TestCase):

    def test_russian(self):
        # Russian words both consisting of Cyrillic
        # and Roman letters can be stemmed.
        stemmer_russian = SnowballStemmer("russian")
        assert stemmer_russian.stem("авантненькая") == "авантненьк"
        assert stemmer_russian.stem("avenantnen'kai^a") == "avenantnen'k"

    def test_german(self):
        stemmer_german = SnowballStemmer("german")
        stemmer_german2 = SnowballStemmer("german", ignore_stopwords=True)

        assert stemmer_german.stem("Schr\xe4nke") == 'schrank'
        assert stemmer_german2.stem("Schr\xe4nke") == 'schrank'

        assert stemmer_german.stem("keinen") == 'kein'
        assert stemmer_german2.stem("keinen") == 'keinen'

    def test_spanish(self):
        stemmer = SnowballStemmer('spanish')

        assert stemmer.stem("Visionado") == 'vision'

        # The word 'algue' was raising an IndexError
        assert stemmer.stem("algue") == 'algu'

    def test_short_strings_bug(self):
        stemmer = SnowballStemmer('english')
        assert stemmer.stem("y's") == 'y'

class PorterTest(unittest.TestCase):
    
    def test_martins_vocabulary(self):
        """Tests all words from the test vocabulary provided by M Porter
        
        The sample vocabulary and output were sourced from:
            http://tartarus.org/martin/PorterStemmer/voc.txt
            http://tartarus.org/martin/PorterStemmer/output.txt
        and are linked to from the Porter Stemmer algorithm's homepage
        at
            http://tartarus.org/martin/PorterStemmer/
        """
        files_folder = os.path.join(os.path.dirname(__file__), 'files')
        words_file = open(os.path.join(files_folder, 'porter_vocabulary.txt'))
        stems_file = open(os.path.join(files_folder, 'porter_output.txt'))
        words = words_file.read().splitlines()
        stems = stems_file.read().splitlines()
        stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
        for word, true_stem in zip(words, stems):
            our_stem = stemmer.stem(word)
            assert our_stem == true_stem, (
                "%s should stem to %s but got %s" % (word, true_stem, our_stem)
            )
