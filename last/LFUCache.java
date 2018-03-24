package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/16/17.
 */
public class LFUCache {

    class Node{
        int freq = 1;
        Node pre;
        Node next;
        Set<Integer> set;
        public Node(){
            set = new LinkedHashSet<>();
        }
    }

    Map<Integer, Integer> vmap;
    Map<Integer, Node> nmap;
    int capacity;
    Node head;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        vmap = new HashMap<>();
        nmap = new HashMap<>();
        head = null;
    }

    public int get(int key) {
        if(vmap.containsKey(key)) {
            increase(key);
            return vmap.get(key);
        }
        return -1;
    }

    public void put(int key, int value) {
        if(capacity == 0) return;
        if(vmap.containsKey(key)){
            vmap.put(key, value);
            increase(key);
        }else{
            if(vmap.size() >= capacity){
                removeLeaseFreqKey();
            }
            vmap.put(key,value);
            addToHead(key);
        }
    }
    private void addToHead(int key) {
        if(head == null) {
            head = new Node();
            head.set.add(key);
        }else if(head.freq > 1){
            Node newNode = new Node();
            newNode.set.add(key);
            newNode.next = head;
            head.pre = newNode;
            newNode.pre =null;
            head = newNode;
        }else{
            head.set.add(key);
        }
        nmap.put(key, head);
    }


    private void removeLeaseFreqKey(){
        if (head == null) return ;
        int old = head.set.iterator().next();
//        for(int n : head.set) {
//            old = n;
//            break;
//        }
        head.set.remove(old);
        if(head.set.size() == 0){
            remove(head);
        }
        vmap.remove(old);
        nmap.remove(old);
    }
    private void remove(Node node){
        if(node.pre != null){
            node.pre.next = node.next;
        }else{
            head = node.next;
        }
        if(node.next != null){
            node.next.pre = node.pre;

        }
    }
    private void increase (int key) {
        Node node = nmap.get(key);
        int freq = node.freq;
        node.set.remove(key);
        if(node.next != null && node.next.freq == freq + 1) {

            node.next.set.add(key);

        }else{
            Node newNode = new Node();
            newNode.freq = node.freq + 1;
            newNode.set.add(key);
            if(node.next != null){
                node.next.pre = newNode;
            }
            newNode.next = node.next;
            node.next = newNode;
            newNode.pre = node;
//             if(node.next == null){
//                 node.next = newNode;
//                 newNode.pre = node;
//             }else{
//                 newNode.next = node.next;
//                 node.next.pre = newNode;
//                 node.next = newNode;
//                 newNode.pre = node;

//             }

        }
        nmap.put(key, node.next);
        if(node.set.size() == 0) remove(node);// remember to remove the empty node;
    }
    // private void insertNext(int key, Node node){
    //     Node newNode = new Node();
    //     newNode.freq = node.freq + 1;
    //     newNode.set.add(key);
    //     node.set.remove(key);
    //     if(node.next != null){
    //         node.next.pre = newNode;
    //     }
    //     newNode.next = node.next;
    //     node.next = newNode;
    //     newNode.pre = node;
    //     nmap.put(key,newNode);
    // }


}

