package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Created by Chang on 9/15/17.
 */
public class leetcode {
    class Node{
        int key;
        int value;
        Node pre;
        Node next;
        public Node (int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
    int capacity;
    Map<Integer, Node> map = new HashMap<>();
    Node head = null;
    Node tail = null;
    public leetcode(int capacity){
        this.capacity = capacity;
    }
    public int get(int key) {
        if(map.containsKey(key)) {
            Node n = map.get(key);
            remove(n);
            setHead(n);
            return n.value;

        }
        return -1;
    }
    private void remove(Node n) {
        if(n.pre != null){
            n.pre.next = n.next;

        }else{
            head = n.next;
        }
        if(n.next != null) {
            n.next = n.pre;
        }else{
            tail = n.pre;
        }
    }

    private void setHead( Node n ) {
        n.next = head;
        n.pre = null;
        if(head != null){
            head.pre = n;

        }
        head = n;
        if(tail == null){
            tail = head;
        }
    }

    public void set(int key, int value) {
        if(map.containsKey(key)) {
            Node n = map.get(key);
            n.value = value;
            remove(n);
            setHead(n);
        }else{
            Node newNode = new Node(key, value);
            if(map.size() >= capacity) {
                map.remove(tail.key);
                remove(tail);

                map.put(key, newNode);
                setHead(newNode);
            }else{
                setHead(newNode);
                map.put(key, newNode);
            }
        }
    }
    class WordFileter {
        TrieNode root;

        public WordFileter(String[] words) {
            root = new TrieNode();
            for (int i = 0; i < words.length; i++) {
                String s = words[i];
                String comp = '{' + s;

                for (int j = s.length(); j >= 0; j--) {
                    insert(s.substring(j) + comp, i);
                }
            }
        }

        public int f(String prefix, String suffix) {
            String toSearch = suffix + '{' + prefix;
            TrieNode cur = root;
            for (char c : toSearch.toCharArray()) {
                cur = root.cld[c - 'a'];
                if (cur == null) return -1;

            }
            return cur.weight;


        }

        private void insert(String s, int idx) {
            TrieNode cur = root;
            for (int i = 0; i < s.length(); i++) {
                if (cur.cld[s.charAt(i) - 'a'] == null) {
                    cur.cld[s.charAt(i) - 'a'] = new TrieNode();
                }
                cur = cur.cld[s.charAt(i) - 'a'];
                cur.weight = idx;
            }
        }


        class TrieNode {
            int weight;
            TrieNode[] cld;

            public TrieNode() {
                cld = new TrieNode[27];
                weight = 0;
            }
        }
    }
}
