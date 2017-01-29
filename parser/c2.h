#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <map>
#include <string>

namespace cpyp {

class Corpus {
 //typedef std::unordered_map<std::string, unsigned, std::hash<std::string> > Map;
// typedef std::unordered_map<unsigned,std::string, std::hash<std::string> > ReverseMap;
public: 
   bool USE_SPELLING=false; 

   std::map<int,std::vector<unsigned>> correct_act_sent;
   std::map<int,std::vector<unsigned>> sentences;
   std::map<int,std::vector<unsigned>> sentencesPos;

   std::map<int,std::vector<unsigned>> correct_act_sentDev;
   std::map<int,std::vector<unsigned>> sentencesDev;
   std::map<int,std::vector<unsigned>> sentencesPosDev;
   std::map<int,std::vector<std::string>> sentencesStrDev;
   unsigned nsentencesDev;

   unsigned nsentences;
   unsigned nwords;
   unsigned nactions;
   unsigned npos;

   unsigned nsentencestest;
   unsigned nsentencesdev;
   int max;
   int maxPos;

   std::map<std::string, unsigned> wordsToInt;
   std::map<unsigned, std::string> intToWords;
   std::vector<std::string> actions;

   std::map<std::string, unsigned> posToInt;
   std::map<unsigned, std::string> intToPos;

   int maxChars;
   std::map<std::string, unsigned> charsToInt;
   std::map<unsigned, std::string> intToChars;

   std::set<unsigned> training_vocab;
   std::set<unsigned> singletons;
   std::map<unsigned, unsigned> counts;

   // String literals
   static constexpr const char* UNK = "UNK";
   static constexpr const char* BAD0 = "<BAD0>";

/*  std::map<unsigned,unsigned>* headsTraining;
  std::map<unsigned,std::string>* labelsTraining;

  std::map<unsigned,unsigned>*  headsParsing;
  std::map<unsigned,std::string>* labelsParsing;*/

 public:
  Corpus() {
    max = 0;
    maxPos = 0;
    maxChars=0; //Miguel
  }


inline unsigned UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else return 0;
}

inline void load_all_act(std::string file){

  std::ifstream actionsFile(file);
  std::string lineS;

  while (getline(actionsFile, lineS)){
    	actions.push_back(lineS);
  }

  nactions=actions.size();

  /* std::cerr<<"nactions:"<<nactions<<"\n"; */
  /* std::cerr<<"done"<<"\n"; */
  /* for (auto a: actions) { */
  /*   std::cerr<<a<<"\n"; */
  /* } */
  actionsFile.close();
}

inline void load_all_pos(std::string file){

  std::ifstream posFile(file);
  std::string lineS;
  assert(maxPos == 0);
  maxPos=1;

  while (getline(posFile, lineS)){
    if (posToInt[lineS] == 0) {
      posToInt[lineS] = maxPos;
      intToPos[maxPos] = lineS;
      npos = maxPos;
      maxPos++;
    }
  }
  posFile.close();
}

inline void load_all_voc(std::string file){

  wordsToInt[Corpus::BAD0] = 0;
  intToWords[0] = Corpus::BAD0;
  wordsToInt[Corpus::UNK] = 1; // unknown symbol
  intToWords[1] = Corpus::UNK;
  assert(max == 0);
  max=2;

  charsToInt[BAD0]=1;
  intToChars[1]="BAD0";
  maxChars=1;

  std::ifstream vocFile(file);
  std::string word;
  unsigned index;
  while (getline(vocFile, word)){
          // new word

    if (wordsToInt[word] == 0) {
      wordsToInt[word] = max;
      intToWords[max] = word;
      nwords = max;
      max++;

      unsigned j = 0;
      while(j < word.length()) {
	std::string wj = "";
	for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
	  wj += word[h];
	}
	if (charsToInt[wj] == 0) {
	  charsToInt[wj] = maxChars;
	  intToChars[maxChars] = wj;
	  maxChars++;
	}
	j += UTF8Len(word[j]);
      }
    }
    index = wordsToInt[word];
    training_vocab.insert(index);
    counts[index]++;

  }
  vocFile.close();
  for (auto wc : counts)
    if (wc.second == 1) singletons.insert(wc.first);
}


inline void load_correct_actions(std::string file){

  correct_act_sent.clear();
  sentences.clear();
  sentencesPos.clear();

  std::ifstream actionsFile(file);
  std::string lineS;

  int count=-1;
  int sentence=-1;
  bool initial=false;
  bool first=true;
  assert(max != 0);
  assert(maxPos != 0);

  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;

  while (getline(actionsFile, lineS)){
    //istringstream iss(line);
    //string lineS;
 		//iss>>lineS;
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");

    if (lineS.empty()) {
      count = 0;
      if (!first) {
	sentences[sentence] = current_sent;
	sentencesPos[sentence] = current_sent_pos;
      }
      sentence++;
      nsentences = sentence;
      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
    }
    else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
        // the initial line in each sentence may look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        lineS = lineS.substr(3, lineS.size() - 4);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);
        do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          if (posIndex == std::string::npos) {
            std::cerr << "cant find the dash in '" << word << "'" << std::endl;
          }
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
	    std::cerr << "NEW_POS:" << pos << std::endl;
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }

          // new word
          if (wordsToInt[word] == 0) {
	    std::cerr << "NEW_WORD:" << word << std::endl;
            wordsToInt[word] = max;
            intToWords[max] = word;
            nwords = max;
            max++;

            unsigned j = 0;
            while(j < word.length()) {
              std::string wj = "";
              for (unsigned h = j; h < j + UTF8Len(word[j]); h++) {
                wj += word[h];
              }
              if (charsToInt[wj] == 0) {
                charsToInt[wj] = maxChars;
                intToChars[maxChars] = wj;
                maxChars++;
              }
              j += UTF8Len(word[j]);
            }
          }

          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
      }
      initial=false;
    }
    else if (count==1){
      int i=0;
      bool found=false;
      for (auto a: actions) {
	if (a==lineS) {
	  std::vector<unsigned> a=correct_act_sent[sentence];
	  a.push_back(i);
	  correct_act_sent[sentence]=a;
	  found=true;
	}
	i++;
      }
      if (!found) {
	actions.push_back(lineS);
	std::vector<unsigned> a=correct_act_sent[sentence];
	a.push_back(actions.size()-1);
	correct_act_sent[sentence]=a;
      }
      count=0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentences[sentence] = current_sent;
    sentencesPos[sentence] = current_sent_pos;
    sentence++;
    nsentences = sentence;
  }

  actionsFile.close();

  std::cerr<<"# of w:"<<nwords<<"\n";
  std::cerr<<"# of a:"<<nactions<<"\n";
  std::cerr<<"# of p:"<<npos<<"\n";
}

inline unsigned get_or_add_word(const std::string& word) {
  unsigned& id = wordsToInt[word];
  if (id == 0) {
    id = max;
    ++max;
    intToWords[id] = word;
    nwords = max;
  }
  return id;
}

inline void load_correct_actionsDev(std::string file) {

  correct_act_sentDev.clear();
  sentencesDev.clear();
  sentencesPosDev.clear();
  sentencesStrDev.clear();
  nsentencesDev = 0;

  std::ifstream actionsFile(file);
  std::string lineS;

  assert(maxPos > 1);
  assert(max > 3);
  int count = -1;
  int sentence = -1;
  bool initial = false;
  bool first = true;
  std::vector<unsigned> current_sent;
  std::vector<unsigned> current_sent_pos;
  std::vector<std::string> current_sent_str;
  while (getline(actionsFile, lineS)) {
    ReplaceStringInPlace(lineS, "-RRB-", "_RRB_");
    ReplaceStringInPlace(lineS, "-LRB-", "_LRB_");
    if (lineS.empty()) {
      // an empty line marks the end of a sentence.
      count = 0;
      if (!first) {
        sentencesDev[sentence] = current_sent;
        sentencesPosDev[sentence] = current_sent_pos;
        sentencesStrDev[sentence] = current_sent_str;
      }

      sentence++;
      nsentencesDev = sentence;

      initial = true;
      current_sent.clear();
      current_sent_pos.clear();
      current_sent_str.clear();
    }
    else if (count == 0) {
      first = false;
      //stack and buffer, for now, leave it like this.
      count = 1;
      if (initial) {
        // the initial line in each sentence may look like:
        // [][the-det, cat-noun, is-verb, on-adp, the-det, mat-noun, ,-punct, ROOT-ROOT]
        // first, get rid of the square brackets.
        lineS = lineS.substr(3, lineS.size() - 4);
        // read the initial line, token by token "the-det," "cat-noun," ...
        std::istringstream iss(lineS);
        do {
          std::string word;
          iss >> word;
          if (word.size() == 0) { continue; }
          // remove the trailing comma if need be.
          if (word[word.size() - 1] == ',') { 
            word = word.substr(0, word.size() - 1);
          }
          // split the string (at '-') into word and POS tag.
          size_t posIndex = word.rfind('-');
          assert(posIndex != std::string::npos);
          std::string pos = word.substr(posIndex + 1);
          word = word.substr(0, posIndex);
          // new POS tag
          if (posToInt[pos] == 0) {
            posToInt[pos] = maxPos;
            intToPos[maxPos] = pos;
            npos = maxPos;
            maxPos++;
          }
          // add an empty string for any token except OOVs (it is easy to 
          // recover the surface form of non-OOV using intToWords(id)).
          current_sent_str.push_back("");
          // OOV word
          if (wordsToInt[word] == 0) {
            if (USE_SPELLING) {
              max = nwords + 1;
              //std::cerr<< "max:" << max << "\n";
              wordsToInt[word] = max;
              intToWords[max] = word;
              nwords = max;
            } else {
              // save the surface form of this OOV before overwriting it.
              current_sent_str[current_sent_str.size()-1] = word;
              word = Corpus::UNK;
            }
          }
          current_sent.push_back(wordsToInt[word]);
          current_sent_pos.push_back(posToInt[pos]);
        } while(iss);
      }
      initial = false;
    }
    else if (count == 1) {
      auto actionIter = std::find(actions.begin(), actions.end(), lineS);
      if (actionIter != actions.end()) {
        unsigned actionIndex = std::distance(actions.begin(), actionIter);
        correct_act_sentDev[sentence].push_back(actionIndex);
      } else {
        // TODO: right now, new actions which haven't been observed in training
        // are not added to correct_act_sentDev. This may be a problem if the
        // training data is little.
      }
      count=0;
    }
  }

  // Add the last sentence.
  if (current_sent.size() > 0) {
    sentencesDev[sentence] = current_sent;
    sentencesPosDev[sentence] = current_sent_pos;
    sentencesStrDev[sentence] = current_sent_str;
    sentence++;
    nsentencesDev = sentence;
  }
  actionsFile.close();
}

void ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

};
}

#endif
