package edu.uci.ics.textdb.exp.dictionarymatcher;
import java.util.*;
/**
 * Created by Chang on 9/18/17.
 */
public class WordLadderii {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        Set<String> dic = new HashSet<>(wordList);
        Map<String, List<String>> map = new HashMap<>();
        List<List<String>> res = new ArrayList<>();
        if(!wordList.contains(endWord)) return res;
        Set<String> start = new HashSet<>();
        start.add(beginWord);
        dic.remove(beginWord);
        boolean found = false;
        while(!start.isEmpty()){
            Set<String> next = new HashSet<>();
            for(String s : start){
                char[] ch = s.toCharArray();
                for(int i = 0; i < s.length(); i++) {
                    char old = ch[i];
                    for(char c = 'a'; c <= 'z'; c++) {
                        if(old == c) continue;
                        ch[i] = c;
                        String ns = String.valueOf(ch);
                        if(dic.contains(ns)){
                            next.add(ns);
                            if(map.containsKey(ns)){
                                map.get(ns).add(s);
                            }else{
                                map.put(ns, new ArrayList<>());
                                map.get(ns).add(s);
                            }
                            if(ns.equals(endWord) && found != true){
                                found = true;
                            }
                        }
                    }
                    ch[i] = old;
                }
            }
            if(found) break;
            dic.removeAll(next);
            start = next;
        }
        List<String> solution = new ArrayList<String>();
        backTrace(map, beginWord, endWord, solution,  res);
        return res;
    }

    private void backTrace(Map<String, List<String>> map, String beginWord, String endWord, List<String> solution, List<List<String>> res) {
        solution.add(0, endWord);
        if (endWord.equals(beginWord)) {
            res.add(new ArrayList<>(solution));
        } else {
            if (map.get(endWord) != null) {


                for (String s : map.get(endWord)) {
                    backTrace(map, beginWord, s, solution, res);
                }
            }
        }
        solution.remove(0);
    }

        public static void main(String[] args) {
        String beginWord = "hit";
        String endWord = "cog";
        List<String> wordList = new ArrayList<>(Arrays.asList("hot","dot","dog","lot","log","cog"));
        List<List<String>> res = new WordLadderii().findLadders(beginWord, endWord, wordList);
        System.out.println(res.size());
        for(List<String> r : res){
            System.out.println(r.toString());
        }
    }

}
