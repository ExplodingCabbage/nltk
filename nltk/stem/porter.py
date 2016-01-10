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

    def _m(self, word, j):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = 0
        while True:
            if i > j:
                return n
            if not self._cons(word, i):
                break
            i = i + 1
        i = i + 1

        while True:
            while True:
                if i > j:
                    return n
                if self._cons(word, i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1

            while True:
                if i > j:
                    return n
                if not self._cons(word, i):
                    break
                i = i + 1
            i = i + 1
            
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

    def _vowelinstem(self, stem):
        """vowelinstem(stem) is TRUE <=> stem contains a vowel"""
        for i in range(len(stem)):
            if not self._cons(stem, i):
                return True
        return False

    def _doublec(self, word):
        """doublec(word) is TRUE <=> word ends with a double consonant"""
        if len(word) < 2:
            return False
        if (word[-1] != word[-2]):
            return False
        return self._cons(word, len(word)-1)

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

    def _cvc(self, word, i):
        """cvc(i) is TRUE <=>

        a) ( --NEW--) i == 1, and word[0] word[1] is vowel consonant, or

        b) word[i - 2], word[i - 1], word[i] has the form consonant -
           vowel - consonant and also if the second c is not w, x or y. this
           is used when trying to restore an e at the end of a short word.
           e.g.

               cav(e), lov(e), hop(e), crim(e), but
               snow, box, tray.
        """
        if i == 0: return False  # i == 0 never happens perhaps
        if i == 1: return (not self._cons(word, 0) and self._cons(word, 1))
        if not self._cons(word, i) or self._cons(word, i-1) or not self._cons(word, i-2): return False

        ch = word[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return False

        return True
        
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
            ('ed', '', self._vowelinstem),  # (*v*) ED  ->   
            ('ing', '', self._vowelinstem), # (*v*) ING ->
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
        """step1c() turns terminal y to i when there is another vowel in the stem.
        --NEW--: This has been modified from the original Porter algorithm so that y->i
        is only done when y is preceded by a consonant, but not if the stem
        is only a single consonant, i.e.

           (*c and not c) Y -> I

        So 'happy' -> 'happi', but
          'enjoy' -> 'enjoy'  etc

        This is a much better rule. Formerly 'enjoy'->'enjoi' and 'enjoyment'->
        'enjoy'. Step 1c is perhaps done too soon; but with this modification that
        no longer really matters.

        Also, the removal of the vowelinstem(z) condition means that 'spy', 'fly',
        'try' ... stem to 'spi', 'fli', 'tri' and conflate with 'spied', 'tried',
        'flies' ...
        """
        if word[-1] == 'y' and len(word) > 2 and self._cons(word, len(word) - 2):
            return word[:-1] + 'i'
        else:
            return word

    def _step2(self, word):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if len(word) <= 1: # Only possible at this stage given unusual inputs to stem_word like 'oed'
            return word

        ch = word[-2]

        if ch == 'a':
            if word.endswith("ational"):
                return word[:-7] + "ate" if self._m(word, len(word)-8) > 0 else word
            elif word.endswith("tional"):
                return word[:-2] if self._m(word, len(word)-7) > 0 else word
            else:
                return word
        elif ch == 'c':
            if word.endswith("enci"):
                return word[:-4] + "ence" if self._m(word, len(word)-5) > 0 else word
            elif word.endswith("anci"):
                return word[:-4] + "ance" if self._m(word, len(word)-5) > 0 else word
            else:
                return word
        elif ch == 'e':
            if word.endswith("izer"):
                return word[:-1] if self._m(word, len(word)-5) > 0 else word
            else:
                return word
        elif ch == 'l':
            if word.endswith("bli"):
                return word[:-3] + "ble" if self._m(word, len(word)-4) > 0 else word # --DEPARTURE--
            # To match the published algorithm, replace "bli" with "abli" and "ble" with "able"
            elif word.endswith("alli"):
                # --NEW--
                if self._m(word, len(word)-5) > 0:
                    word = word[:-2]
                    return self._step2(word)
                else:
                    return word
            elif word.endswith("fulli"):
                return word[:-2] if self._m(word, len(word)-6) else word # --NEW--
            elif word.endswith("entli"):
                return word[:-2] if self._m(word, len(word)-6) else word
            elif word.endswith("eli"):
                return word[:-2] if self._m(word, len(word)-4) else word
            elif word.endswith("ousli"):
                return word[:-2] if self._m(word, len(word)-6) else word
            else:
                return word
        elif ch == 'o':
            if word.endswith("ization"):
                return word[:-7] + "ize" if self._m(word, len(word)-8) else word
            elif word.endswith("ation"):
                return word[:-5] + "ate" if self._m(word, len(word)-6) else word
            elif word.endswith("ator"):
                return word[:-4] + "ate" if self._m(word, len(word)-5) else word
            else:
                return word
        elif ch == 's':
            if word.endswith("alism"):
                return word[:-3] if self._m(word, len(word)-6) else word
            elif word.endswith("ness"):
                if word.endswith("iveness"):
                    return word[:-4] if self._m(word, len(word)-8) else word
                elif word.endswith("fulness"):
                    return word[:-4] if self._m(word, len(word)-8) else word
                elif word.endswith("ousness"):
                    return word[:-4] if self._m(word, len(word)-8) else word
                else:
                    return word
            else:
                return word
        elif ch == 't':
            if word.endswith("aliti"):
                return word[:-3] if self._m(word, len(word)-6) else word
            elif word.endswith("iviti"):
                return word[:-5] + "ive" if self._m(word, len(word)-6) else word
            elif word.endswith("biliti"):
                return word[:-6] + "ble" if self._m(word, len(word)-7) else word
            else:
                return word
        elif ch == 'g': # --DEPARTURE--
            if word.endswith("logi"):
                return word[:-1] if self._m(word, len(word) - 4) else word # --NEW-- (Barry Wilkins)
            # To match the published algorithm, pass len(word)-5 to _m instead of len(word)-4
            else:
                return word

        else:
            return word

    def _step3(self, word):
        """step3() deals with -ic-, -full, -ness etc. similar strategy to step2."""

        ch = word[-1]

        if ch == 'e':
            if word.endswith("icate"):
                return word[:-3] if self._m(word, len(word)-6) else word
            elif word.endswith("ative"):
                return word[:-5] if self._m(word, len(word)-6) else word
            elif word.endswith("alize"):
                return word[:-3] if self._m(word, len(word)-6) else word
            else:
                return word
        elif ch == 'i':
            if word.endswith("iciti"):
                return word[:-3] if self._m(word, len(word)-6) else word
            else:
                return word
        elif ch == 'l':
            if word.endswith("ical"):
                return word[:-2] if self._m(word, len(word)-5) else word
            elif word.endswith("ful"):
                return word[:-3] if self._m(word, len(word)-4) else word
            else:
                return word
        elif ch == 's':
            if word.endswith("ness"):
                return word[:-4] if self._m(word, len(word)-5) else word
            else:
                return word

        else:
            return word

    def _step4(self, word):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""

        if len(word) <= 1: # Only possible at this stage given unusual inputs to stem_word like 'oed'
            return word

        ch = word[-2]

        if ch == 'a':
            if word.endswith("al"):
                return word[:-2] if self._m(word, len(word)-3) > 1 else word
            else:
                return word
        elif ch == 'c':
            if word.endswith("ance"):
                return word[:-4] if self._m(word, len(word)-5) > 1 else word
            elif word.endswith("ence"):
                return word[:-4] if self._m(word, len(word)-5) > 1 else word
            else:
                return word
        elif ch == 'e':
            if word.endswith("er"):
                return word[:-2] if self._m(word, len(word)-3) > 1 else word
            else:
                return word
        elif ch == 'i':
            if word.endswith("ic"):
                return word[:-2] if self._m(word, len(word)-3) > 1 else word
            else:
                return word
        elif ch == 'l':
            if word.endswith("able"):
                return word[:-4] if self._m(word, len(word)-5) > 1 else word
            elif word.endswith("ible"):
                return word[:-4] if self._m(word, len(word)-5) > 1 else word
            else:
                return word
        elif ch == 'n':
            if word.endswith("ant"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            elif word.endswith("ement"):
                return word[:-5] if self._m(word, len(word)-6) > 1 else word
            elif word.endswith("ment"):
                return word[:-4] if self._m(word, len(word)-5) > 1 else word
            elif word.endswith("ent"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        elif ch == 'o':
            if word.endswith("sion") or word.endswith("tion"): # slightly different logic to all the other cases
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            elif word.endswith("ou"):
                return word[:-2] if self._m(word, len(word)-3) > 1 else word
            else:
                return word
        elif ch == 's':
            if word.endswith("ism"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        elif ch == 't':
            if word.endswith("ate"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            elif word.endswith("iti"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        elif ch == 'u':
            if word.endswith("ous"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        elif ch == 'v':
            if word.endswith("ive"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        elif ch == 'z':
            if word.endswith("ize"):
                return word[:-3] if self._m(word, len(word)-4) > 1 else word
            else:
                return word
        else:
            return word

    def _step5(self, word):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        if word[-1] == 'e':
            a = self._m(word, len(word)-1)
            if a > 1 or (a == 1 and not self._cvc(word, len(word)-2)):
                word = word[:-1]
        if word.endswith('ll') and self._m(word, len(word)-1) > 1:
            word = word[:-1]

        return word

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
        stem = self._step5(stem)
        
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
