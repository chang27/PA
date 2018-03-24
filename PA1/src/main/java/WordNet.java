/**
 * Created by Chang on 10/29/17.
 */

import edu.princeton.cs.algs4.*;
import edu.princeton.cs.introcs.In;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WordNet {

//    private Map<Integer, String> map1;
//    private Map<String, Bag<Integer>> map2;
//    private Digraph graph;
//
//    // constructor takes the name of the two input files
//    public WordNet(String synsets, String hypernyms){
//        map1 = new HashMap<>();
//        map2 = new HashMap<>();
//        readS(synsets);
//        graph = readH(hypernyms);
//    }
//
//    private void readS (String synsets) {
//
//        In input = new In(synsets);
//        Bag<Integer> bag; //bag is a linkedlist;
//
//        while (input.hasNextLine()) {
//            String[] list = input.readLine().split(",");
//            int id = Integer.parseInt(list[0]);
//            map1.put(id, list[1]);
//            String[] list2 = list[1].split(" ");
//            for (String str : list2) {
//                if (!map2.containsKey(str)) {
//                    map2.put(str, new Bag<Integer>());
//                }
//                map2.get(str).add(id);
//            }
//        }
//    }
//
//    private Digraph readH(String hypernyms) {
//        In input = new In(hypernyms);
//        Digraph graph = new Digraph(map1.size());
//        while(input.hasNextLine()){
//            String[] nodes = input.readLine().split(",");
//            for(int i = 1; i < nodes.length; i++) {
//                graph.addEdge(Integer.parseInt(nodes[0]), Integer.parseInt(nodes[i]));
//            }
//
//        }
//        DirectedCycle directedCycle = new DirectedCycle(graph);
//            if(directedCycle.hasCycle()){
//                throw new IllegalArgumentException();
//            }
//        int root = 0;
//        for(int i = 0; i < graph.V(); i++){
//                if(!graph.adj(i).iterator().hasNext()){
//                    root++;
//                }
//        }
//        if(root != 1) throw new IllegalArgumentException();
//        return graph;
//    }
//
//    // returns all WordNet nouns
//    public Iterable<String> nouns(){
//            return map2.keySet();
//    }
//
//    // is the word a WordNet noun?
//    public boolean isNoun(String word){
//        if(word == null || word.length() == 0) return false;
//        return map2.containsKey(word);
//    }
//
//    // distance between nounA and nounB (defined below)
//    public int distance(String nounA, String nounB){
//
//    }
//
//    // a synset (second field of synsets.txt) that is the common ancestor of nounA and nounB
//    // in a shortest ancestral path (defined below)
//    public String sap(String nounA, String nounB){
//
//    }
//
//    // do unit testing of this class
//    public static void main(String[] args){
//
//    }

}
