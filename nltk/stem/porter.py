# Additional modifications were made to incorporate this module into
# NLTK.  All such modifications are marked with "--NLTK--".  The NLTK
# version of this module is maintained by NLTK developers,
# and is available via http://nltk.org/

"""
Porter Stemmer

This is the Porter stemming algorithm, ported to Python from the
version coded up in ANSI C by the author. It follows the algorithm
presented in

Porter, M. "An algorithm for suffix stripping." Program 14.3 (1980): 130-137.

only differing from it at the points marked --DEPARTURE-- and --NEW--
below.

For a more faithful version of the Porter algorithm, see

    http://www.tartarus.org/~martin/PorterStemmer/

Later additions:

   June 2000

   The 'l' of the 'logi' -> 'log' rule is put with the stem, so that
   short stems like 'geo' 'theo' etc work like 'archaeo' 'philo' etc.

   This follows a suggestion of Barry Wilkins, research student at
   Birmingham.


   February 2000

   the cvc test for not dropping final -e now looks after vc at the
   beginning of a word, so are, eve, ice, ore, use keep final -e. In this
   test c is any consonant, including w, x and y. This extension was
   suggested by Chris Emerson.

   -fully    -> -ful   treated like  -fulness -> -ful, and
   -tionally -> -tion  treated like  -tional  -> -tion

   both in Step 2. These were suggested by Hiranmay Ghosh, of New Delhi.

   Invariants proceed, succeed, exceed. Also suggested by Hiranmay Ghosh.

Additional modifications were made to incorperate this module into
nltk.  All such modifications are marked with \"--NLTK--\".
"""

from __future__ import print_function, unicode_literals

## --NLTK--
## Declare this module's documentation format.
__docformat__ = 'plaintext'

import re

from nltk.stem.api import StemmerI
from nltk.compat import python_2_unicode_compatible

class _CannotReplaceSuffix(Exception):
    pass

@python_2_unicode_compatible
class PorterStemmer(StemmerI):

    ## --NLTK--
    ## Add a module docstring
    """
    A word stemmer based on the Porter stemming algorithm.

        Porter, M. \"An algorithm for suffix stripping.\"
        Program 14.3 (1980): 130-137.

    A few minor modifications have been made to Porter's basic
    algorithm.  See the source code of this module for more
    information.

    The Porter Stemmer requires that all tokens have string types.
    """

    # The main part of the stemming algorithm starts here.
    # Note that only lower case sequences are stemmed. Forcing to lower case
    # should be done before stem(...) is called.

    def __init__(self):

        ## --NEW--
        ## This is a table of irregular forms. It is quite short, but still
        ## reflects the errors actually drawn to Martin Porter's attention over
        ## a 20 year period!
        ##
        ## Extend it as necessary.
        ##
        ## The form of the table is:
        ##  {
        ##  "p1" : ["s11","s12","s13", ... ],
        ##  "p2" : ["s21","s22","s23", ... ],
        ##  ...
        ##  "pn" : ["sn1","sn2","sn3", ... ]
        ##  }
        ##
        ## String sij is mapped to paradigm form pi, and the main stemming
        ## process is then bypassed.

        irregular_forms = {
            "sky" :     ["sky", "skies"],
            "die" :     ["dying"],
            "lie" :     ["lying"],
            "tie" :     ["tying"],
            "news" :    ["news"],
            "inning" :  ["innings", "inning"],
            "outing" :  ["outings", "outing"],
            "canning" : ["cannings", "canning"],
            "howe" :    ["howe"],

            # --NEW--
            "proceed" : ["proceed"],
            "exceed"  : ["exceed"],
            "succeed" : ["succeed"], # Hiranmay Ghosh
            }

        self.pool = {}
        for key in irregular_forms:
            for val in irregular_forms[key]:
                self.pool[val] = key

        self.vowels = frozenset(['a', 'e', 'i', 'o', 'u'])

    def _cons(self, word, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            else:
                return (not self._cons(word, i - 1))
        return True
        
    def _measure(self, stem):
        """Returns the 'measure' of stem, per definition in the paper
        
        From the paper:
        
            A consonant will be denoted by c, a vowel by v. A list
            ccc... of length greater than 0 will be denoted by C, and a
            list vvv... of length greater than 0 will be denoted by V.
            Any word, or part of a word, therefore has one of the four
            forms:

                CVCV ... C
                CVCV ... V
                VCVC ... C
                VCVC ... V
                
            These may all be represented by the single form
            
                [C]VCVC ... [V]
                
            where the square brackets denote arbitrary presence of their
            contents. Using (VC){m} to denote VC repeated m times, this
            may again be written as

                [C](VC){m}[V].

            m will be called the \measure\ of any word or word part when
            represented in this form. The case m = 0 covers the null
            word. Here are some examples:

                m=0    TR,  EE,  TREE,  Y,  BY.
                m=1    TROUBLE,  OATS,  TREES,  IVY.
                m=2    TROUBLES,  PRIVATE,  OATEN,  ORRERY.
        """
        cv_sequence = ''
        
        # Construct a string of 'c's and 'v's representing whether each
        # character in `stem` is a consonsant or a vowel.
        # e.g. 'falafel' becomes 'cvcvcvc',
        #      'architecture' becomes 'vcccvcvccvcv'
        for i in range(len(stem)):
            if self._cons(stem, i):
                cv_sequence += 'c'
            else:
                cv_sequence += 'v'
                
        # Count the number of 'vc' occurences, which is equivalent to
        # the number of 'VC' occurrences in Porter's reduced form in the
        # docstring above, which is in turn equivalent to `m`
        return cv_sequence.count('vc')
        
    def _has_positive_measure(self, stem):
        return self._measure(stem) > 0

    def _contains_vowel(self, stem):
        """_contains_vowel(stem) is TRUE <=> stem contains a vowel"""
        for i in range(len(stem)):
            if not self._cons(stem, i):
                return True
        return False

    def _ends_cvc(self, word):
        """Implements condition *o from the paper
        
        From the paper:
        
            *o  - the stem ends cvc, where the second c is not W, X or Y
                  (e.g. -WIL, -HOP).
        """
        return (
            len(word) >= 3 and
            self._cons(word, len(word) - 3) and
            not self._cons(word, len(word) - 2) and
            self._cons(word, len(word) - 1) and
            word[-1] not in ('w', 'x', 'y')
        )
        
    def _replace_suffix(self, word, suffix, replacement):
        """Replaces `suffix` of `word` with `replacement"""
        assert word.endswith(suffix), "Given word doesn't end with given suffix"
        return word[:-len(suffix)] + replacement

    def _replace_suffix_if(self, word, suffix, replacement, condition):
        """If `condition`, replace suffix with replacement, else raise
        
        `condition` should be a lambda that takes the word and stem as
        arguments and returns True or False.
        """
        if not word.endswith(suffix):
            raise _CannotReplaceSuffix("word does not end with suffix")
        else:
            stem = self._replace_suffix(word, suffix, replacement)
            if condition is None or condition(stem):
                return stem
            else:
                raise _CannotReplaceSuffix("condition not met")
                
    def _apply_first_possible_rule(self, word, rules):
        """Applies the first applicable suffix-removal rule to the word
        
        Takes a word and a list of suffix-removal rules represented as
        3-tuples, with the first element being the suffix to remove,
        the second element being the string to replace it with, and the
        final element being the condition for the rule to be applicable,
        or None if the rule is unconditional.
        """
        for rule in rules:
            try:
                return self._replace_suffix_if(word, *rule)
            except _CannotReplaceSuffix:
                pass
                
        return word
        
    def _step1a(self, word):
        """Implements Step 1a from "An algorithm for suffix stripping"
        
        From the paper:
            
            SSES -> SS                         caresses  ->  caress
            IES  -> I                          ponies    ->  poni
                                               ties      ->  ti
            SS   -> SS                         caress    ->  caress
            S    ->                            cats      ->  cat
        """
        return self._apply_first_possible_rule(word, [
            ('sses', 'ss', None), # SSES -> SS
            
            # --NLTK--
            # this line extends the original algorithm, so that
            # 'flies'->'fli' but 'dies'->'die' etc
            ('ies', 'ie', lambda stem: len(word) == 4),
            
            ('ies', 'i', None),   # IES  -> I
            ('ss', 'ss', None),   # SS   -> SS
            ('s', '', None),      # S    ->
        ])
        
    def _step1b(self, word):
        """Implements Step 1b from "An algorithm for suffix stripping"
        
        From the paper:
        
            (m>0) EED -> EE                    feed      ->  feed
                                               agreed    ->  agree
            (*v*) ED  ->                       plastered ->  plaster
                                               bled      ->  bled
            (*v*) ING ->                       motoring  ->  motor
                                               sing      ->  sing
                                               
        If the second or third of the rules in Step 1b is successful, the following
        is done:

            AT -> ATE                       conflat(ed)  ->  conflate
            BL -> BLE                       troubl(ed)   ->  trouble
            IZ -> IZE                       siz(ed)      ->  size
            (*d and not (*L or *S or *Z))
               -> single letter
                                            hopp(ing)    ->  hop
                                            tann(ed)     ->  tan
                                            fall(ing)    ->  fall
                                            hiss(ing)    ->  hiss
                                            fizz(ed)     ->  fizz
            (m=1 and *o) -> E               fail(ing)    ->  fail
                                            fil(ing)     ->  file

        The rule to map to a single letter causes the removal of one of the double
        letter pair. The -E is put back on -AT, -BL and -IZ, so that the suffixes
        -ATE, -BLE and -IZE can be recognised later. This E may be removed in step
        4.
        """
        # --NLTK-- 
        # this block extends the original algorithm, so that
        # 'spied'->'spi' but 'died'->'die' etc
        try:
            return self._replace_suffix_if(
                word, 'ied', 'ie', lambda stem: len(word) == 4
            )
        except _CannotReplaceSuffix:
            pass
        
        try:
            # (m>0) EED -> EE
            return self._replace_suffix_if(
                word, 'eed', 'ee', lambda stem: self._measure(stem) > 0
            )
        except _CannotReplaceSuffix:
            pass
            
        rule_2_or_3_succeeded = False
        for rule in [
            ('ed', '', self._contains_vowel),  # (*v*) ED  ->   
            ('ing', '', self._contains_vowel), # (*v*) ING ->
        ]:
            try:
                intermediate_stem = self._replace_suffix_if(word, *rule)
                rule_2_or_3_succeeded = True
                break
            except _CannotReplaceSuffix:
                pass
                
        if not rule_2_or_3_succeeded:
            return word
        
        final_letter = intermediate_stem[-1]
        return self._apply_first_possible_rule(intermediate_stem, [
            ('at', 'ate', None), # AT -> ATE
            ('bl', 'ble', None), # BL -> BLE
            ('iz', 'ize', None), # IZ -> IZE
            # (*d and not (*L or *S or *Z))
            # -> single letter
            (
                final_letter*2,
                final_letter,
                lambda stem: final_letter not in ('l', 's', 'z')
            ),
            # (m=1 and *o) -> E
            (
                '',
                'e',
                lambda stem: (self._measure(stem) == 1 and
                              self._ends_cvc(stem))
            ),
        ])
    
    def _step1c(self, word):
        """Implements Step 1c from "An algorithm for suffix stripping"
        
        From the paper:
        
        Step 1c

            (*v*) Y -> I                    happy        ->  happi
                                            sky          ->  sky
        
        --NEW--: This has been modified from the original Porter algorithm so that y->i
        is only done when y is preceded by a consonant, but not if the stem
        is only a single consonant, i.e.

           (*c and not c) Y -> I

        So 'happy' -> 'happi', but
          'enjoy' -> 'enjoy'  etc

        This is a much better rule. Formerly 'enjoy'->'enjoi' and 'enjoyment'->
        'enjoy'. Step 1c is perhaps done too soon; but with this modification that
        no longer really matters.

        Also, the removal of the contains_vowel(z) condition means that 'spy', 'fly',
        'try' ... stem to 'spi', 'fli', 'tri' and conflate with 'spied', 'tried',
        'flies' ...
        """
        try:
            return self._replace_suffix_if(
                word,
                'y',
                'i',
                lambda stem: len(word) > 2 and self._cons(word, len(word) - 2)
            )
        except _CannotReplaceSuffix:
            return word

    def _step2(self, word):
        """Implements Step 2 from "An algorithm for suffix stripping"
        
        From the paper:
        
        Step 2

            (m>0) ATIONAL ->  ATE       relational     ->  relate
            (m>0) TIONAL  ->  TION      conditional    ->  condition
                                        rational       ->  rational
            (m>0) ENCI    ->  ENCE      valenci        ->  valence
            (m>0) ANCI    ->  ANCE      hesitanci      ->  hesitance
            (m>0) IZER    ->  IZE       digitizer      ->  digitize
            (m>0) ABLI    ->  ABLE      conformabli    ->  conformable
            (m>0) ALLI    ->  AL        radicalli      ->  radical
            (m>0) ENTLI   ->  ENT       differentli    ->  different
            (m>0) ELI     ->  E         vileli        - >  vile
            (m>0) OUSLI   ->  OUS       analogousli    ->  analogous
            (m>0) IZATION ->  IZE       vietnamization ->  vietnamize
            (m>0) ATION   ->  ATE       predication    ->  predicate
            (m>0) ATOR    ->  ATE       operator       ->  operate
            (m>0) ALISM   ->  AL        feudalism      ->  feudal
            (m>0) IVENESS ->  IVE       decisiveness   ->  decisive
            (m>0) FULNESS ->  FUL       hopefulness    ->  hopeful
            (m>0) OUSNESS ->  OUS       callousness    ->  callous
            (m>0) ALITI   ->  AL        formaliti      ->  formal
            (m>0) IVITI   ->  IVE       sensitiviti    ->  sensitive
            (m>0) BILITI  ->  BLE       sensibiliti    ->  sensible
        """

        # --NEW--
        # Instead of applying the ALLI -> AL rule after 'bli' per the
        # published algorithm, instead we apply it first, and, if it
        # succeeds, run the result through step2 again.
        try:
            stem = self._replace_suffix_if(
                word,
                'alli',
                'al',
                self._has_positive_measure
            )
            return self._step2(stem)
        except _CannotReplaceSuffix:
            pass
        
        return self._apply_first_possible_rule(word, [
            ('ational', 'ate', self._has_positive_measure),
            ('tional', 'tion', self._has_positive_measure),
            ('enci', 'ence', self._has_positive_measure),
            ('anci', 'ance', self._has_positive_measure),
            ('izer', 'ize', self._has_positive_measure),
            
            # --DEPARTURE--
            # To match the published algorithm, replace "bli" with
            # "abli" and "ble" with "able"
            ('bli', 'ble', self._has_positive_measure),
            
            # -- NEW --
            ('fulli', 'ful', self._has_positive_measure),
            
            ('entli', 'ent', self._has_positive_measure),
            ('eli', 'e', self._has_positive_measure),
            ('ousli', 'ous', self._has_positive_measure),
            ('ization', 'ize', self._has_positive_measure),
            ('ation', 'ate', self._has_positive_measure),
            ('ator', 'ate', self._has_positive_measure),
            ('alism', 'al', self._has_positive_measure),
            ('iveness', 'ive', self._has_positive_measure),
            ('fulness', 'ful', self._has_positive_measure),
            ('ousness', 'ous', self._has_positive_measure),
            ('aliti', 'al', self._has_positive_measure),
            ('iviti', 'ive', self._has_positive_measure),
            ('biliti', 'ble', self._has_positive_measure),
            
            # --DEPARTURE--
            # To match the published algorithm, delete this phrase
            # --NEW-- (Barry Wilkins)
            # To match the published algorithm, replace lambda below
            # with just self._has_positive_measure
            ("logi", "log", lambda stem: self._has_positive_measure(word[:-3])),
        ])

    def _step3(self, word):
        """Implements Step 3 from "An algorithm for suffix stripping"
        
        From the paper:
        
        Step 3

            (m>0) ICATE ->  IC              triplicate     ->  triplic
            (m>0) ATIVE ->                  formative      ->  form
            (m>0) ALIZE ->  AL              formalize      ->  formal
            (m>0) ICITI ->  IC              electriciti    ->  electric
            (m>0) ICAL  ->  IC              electrical     ->  electric
            (m>0) FUL   ->                  hopeful        ->  hope
            (m>0) NESS  ->                  goodness       ->  good
        """
        return self._apply_first_possible_rule(word, [
            ('icate', 'ic', self._has_positive_measure),
            ('ative', '', self._has_positive_measure),
            ('alize', 'al', self._has_positive_measure),
            ('iciti', 'ic', self._has_positive_measure),
            ('ical', 'ic', self._has_positive_measure),
            ('ful', '', self._has_positive_measure),
            ('ness', '', self._has_positive_measure),
        ])

    def _step4(self, word):
        """Implements Step 4 from "An algorithm for suffix stripping"
        
        Step 4

            (m>1) AL    ->                  revival        ->  reviv
            (m>1) ANCE  ->                  allowance      ->  allow
            (m>1) ENCE  ->                  inference      ->  infer
            (m>1) ER    ->                  airliner       ->  airlin
            (m>1) IC    ->                  gyroscopic     ->  gyroscop
            (m>1) ABLE  ->                  adjustable     ->  adjust
            (m>1) IBLE  ->                  defensible     ->  defens
            (m>1) ANT   ->                  irritant       ->  irrit
            (m>1) EMENT ->                  replacement    ->  replac
            (m>1) MENT  ->                  adjustment     ->  adjust
            (m>1) ENT   ->                  dependent      ->  depend
            (m>1 and (*S or *T)) ION ->     adoption       ->  adopt
            (m>1) OU    ->                  homologou      ->  homolog
            (m>1) ISM   ->                  communism      ->  commun
            (m>1) ATE   ->                  activate       ->  activ
            (m>1) ITI   ->                  angulariti     ->  angular
            (m>1) OUS   ->                  homologous     ->  homolog
            (m>1) IVE   ->                  effective      ->  effect
            (m>1) IZE   ->                  bowdlerize     ->  bowdler

        The suffixes are now removed. All that remains is a little
        tidying up.
        """
        measure_gt_1 = lambda stem: self._measure(stem) > 1
        
        return self._apply_first_possible_rule(word, [
            ('al', '', measure_gt_1),
            ('ance', '', measure_gt_1),
            ('ence', '', measure_gt_1),
            ('er', '', measure_gt_1),
            ('ic', '', measure_gt_1),
            ('able', '', measure_gt_1),
            ('ible', '', measure_gt_1),
            ('ant', '', measure_gt_1),
            ('ement', '', measure_gt_1),
            ('ment', '', measure_gt_1),
            ('ent', '', measure_gt_1),
            
            # (m>1 and (*S or *T)) ION -> 
            (
                'ion',
                '',
                lambda stem: self._measure(stem) > 1 and stem[-1] in ('s', 't')
            ),
            
            ('ou', '', measure_gt_1),
            ('ism', '', measure_gt_1),
            ('ate', '', measure_gt_1),
            ('iti', '', measure_gt_1),
            ('ous', '', measure_gt_1),
            ('ive', '', measure_gt_1),
            ('ize', '', measure_gt_1),
        ])
        
    def _step5a(self, word):
        """Implements Step 5a from "An algorithm for suffix stripping"
        
        From the paper:
        
        Step 5a

            (m>1) E     ->                  probate        ->  probat
                                            rate           ->  rate
            (m=1 and not *o) E ->           cease          ->  ceas
        """
        return self._apply_first_possible_rule(word, [
            ('e', '', lambda stem: self._measure(stem) > 1),
            (
                'e',
                '',
                lambda stem: (
                    self._measure(stem) == 1 and
                    not self._ends_cvc(stem)
                )
            )
        ])

    def _step5b(self, word):
        """Implements Step 5a from "An algorithm for suffix stripping"
        
        From the paper:
        
        Step 5b

            (m > 1 and *d and *L) -> single letter
                                    controll       ->  control
                                    roll           ->  roll
        """
        # The rule is expressed in an overcomplicated way in Porter's
        # paper, but all it means it that double-l should become
        # single-l. It could've been written more straightforwardly as:
        #
        #     (m > 1) LL -> L
        return self._apply_first_possible_rule(word, [
            ('ll', 'l', lambda stem: self._measure(stem) > 1)
        ])

    def stem(self, word):
        stem = word.lower()
        
        # --NLTK--
        if word in self.pool:
            return self.pool[word]

        if len(word) <= 2:
            return word # --DEPARTURE--
        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        stem = self._step1a(stem)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)
        
        return stem

    def __repr__(self):
        return '<PorterStemmer>'

def demo():
    """
    A demonstration of the porter stemmer on a sample from
    the Penn Treebank corpus.
    """

    from nltk.corpus import treebank
    from nltk import stem

    stemmer = stem.PorterStemmer()

    orig = []
    stemmed = []
    for item in treebank.files()[:3]:
        for (word, tag) in treebank.tagged_words(item):
            orig.append(word)
            stemmed.append(stemmer.stem(word))

    # Convert the results to a string, and word-wrap them.
    results = ' '.join(stemmed)
    results = re.sub(r"(.{,70})\s", r'\1\n', results+' ').rstrip()

    # Convert the original to a string, and word wrap it.
    original = ' '.join(orig)
    original = re.sub(r"(.{,70})\s", r'\1\n', original+' ').rstrip()

    # Print the results.
    print('-Original-'.center(70).replace(' ', '*').replace('-', ' '))
    print(original)
    print('-Results-'.center(70).replace(' ', '*').replace('-', ' '))
    print(results)
    print('*'*70)
