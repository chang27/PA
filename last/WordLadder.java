package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/18/17.
 */
public class WordLadder {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
       Set<String> dic = new HashSet<>(wordList);
       Set<String> begin = new HashSet<>();
       begin.add(beginWord);
       Set<String> end = new HashSet<>();
       end.add(endWord);
       int len = 2;
       dic.remove(beginWord);
       dic.remove(endWord);
       while(! begin.isEmpty() && ! end.isEmpty()){
           if(begin.size() > end.size()){
               Set<String> tmp = begin;
               begin = end;
               end = tmp;
           }
           Set<String> next = new HashSet<>();
           for(String s : begin){
               char[] ch = s.toCharArray();
               for(int i = 0; i < ch.length; i++){
                   char old = ch[i];
                   for(char c = 'a'; c < 'z'; c++){
                      if(ch[i] == c) continue;
                      ch[i] = c;
                      String ns = String.valueOf(ch);
                      if(end.contains(ns)) return len;
                      if(dic.contains(ns)){
                          next.add(ns);
                          dic.remove(ns);
                      }

                   }
                   ch[i] = old;
               }
           }
           len++;
           begin = next;
       }
       return 0;
    }
}
