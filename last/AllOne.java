package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by Chang on 9/17/17.
 */
public class AllOne {
    class Node{
        Node pre;
        Node next;
        int val;
        Set<String> set;
        public Node(int val) {
            this.val = val;
            set = new LinkedHashSet<>();
        }
    }

    Map<String, Integer> vmap;
    Map<Integer, Node> nmap;
    Node head = null;
    Node tail = null;

    public AllOne() {
        vmap = new HashMap<>();
        nmap = new HashMap<>();
    }

    public void inc(String key) {
        if(vmap.containsKey(key)){
            int val = vmap.get(key);
            increase(key, val);
            vmap.put(key, val + 1);
        }else{
            vmap.put(key, 1);
            add(key);
        }
    }

    private void increase(String key, int val) {
        Node node = nmap.get(val);
            node.set.remove(key);
        if(node.next != null && node.next.val == val + 1){
            node.next.set.add(key);
        }else{
            Node newNode = new Node(val + 1);
            newNode.set.add(key);
            if(node.next != null) {
                newNode.next = node.next;
                node.next.pre = newNode;
                node.next = newNode;
                newNode.pre = node;
                nmap.put(val+1, newNode);
            }else{
                node.next = newNode;
                newNode.pre = node;
                tail = newNode;

            }
            nmap.put(val+1, newNode);
        }
        if(node.set.isEmpty()) remove(node);
    }
    private void add(String key){
        if(head == null){
            head = new Node(1);
            head.set.add(key);
            nmap.put(1, head);
        }else if(head.val > 1){
            Node node = new Node(1);
            node.set.add(key);
            node.next = head;
            head.pre = node;
            node.pre = null;
            head = node;
            nmap.put(1, head);
        }else{
            head.set.add(key);
        }
        if(tail == null){
            tail = head;
        }

    }
    public void dec(String key) {
        if(! vmap.containsKey(key)) return;
        int val = vmap.get(key);
        if(val == 1) {
            vmap.remove(key);
        } else{
            vmap.put(key, val -1);
        }
        decrease(key, val);
    }
    private void decrease(String key, int val){
        if(val == 1){
            head.set.remove(key);
            if(head.set.isEmpty()){
                remove(head);
            }
        }else{
            Node node = nmap.get(val);
            node.set.remove(key);
            if(node.pre != null && node.pre.val == val - 1){
                node.pre.set.add(key);
            }else{
                Node newNode = new Node(val - 1);
                newNode.set.add(key);
                if(node.pre != null) {
                    newNode.next = node;
                    newNode.pre = node.pre;
                    node.pre.next = newNode;
                    node.pre = newNode;
                }else{
                    node.pre = newNode;
                    newNode.next = node;
                    newNode.pre = null;
                    head = newNode;
                }
                nmap.put(val - 1, newNode);
            }
            if(node.set.isEmpty()){
                remove(node);
            }
        }

    }
    private void remove(Node node){
        if(node.pre != null){
            node.pre.next = node.next;
        }else{
            head = node.next;
        }

        if(node.next != null){
            node.next.pre = node.pre;

        }else{
            tail = node.pre;
        }
    }
    public String getMaxKey() {
        if(tail == null) return "";
        return tail.set.iterator().next();
    }

    /** Returns one of the keys with Minimal value. */
    public String getMinKey() {
        if(head == null) return "";
        return head.set.iterator().next();

    }

    public static void main(String[] args){
        AllOne ao = new AllOne();
        ao.inc("hello");
       ao.inc("hello");
       ao.inc("world");
       ao.inc("world");
       ao.inc("hello");

      //  System.out.println(ao.getMaxKey());
        ao.dec("world");
        System.out.println(ao.getMinKey());
        System.out.println(ao.getMaxKey());
        ao.inc("world");
        ao.inc("world");
        ao.inc("leet");
        System.out.println(ao.getMinKey());
        System.out.println(ao.getMaxKey());
        ao.inc("leet");

        ao.inc("leet");
        System.out.println(ao.getMinKey());
       // System.out.println(ao.getMaxKey());
    }
}
