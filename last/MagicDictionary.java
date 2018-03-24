package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/30/17.
 */

 public  class MagicDictionary {
        Map<String, List<Character>> map;
        /** Initialize your data structure here. */
        public MagicDictionary() {
            map = new HashMap<>();
        }

        /** Build a dictionary through a list of words */
        public void buildDict(String[] dict) {
            for(String s :dict){
                StringBuilder sb = new StringBuilder(s);
                for(int i = 0; i < s.length(); i++) {
                    sb.setCharAt(i, '*');
                    if(!map.containsKey(sb.toString())) {
                        map.put(sb.toString(), new ArrayList<>());
                    }
                    map.get(sb.toString()).add(s.charAt(i));
                    sb.setCharAt(i, s.charAt(i));
                }
            }
        }

        /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
        public boolean search(String word) {
            StringBuilder sb = new StringBuilder(word);
            for(int i = 0; i < word.length(); i++) {
                sb.setCharAt(i, '*');
                if(map.containsKey(sb.toString()) && !map.get(sb.toString()).contains(word.charAt(i))){
                    return true;
                }
                sb.setCharAt(i, word.charAt(i));
            }
            return false;
        }

        public static void main(String[] args){
            MagicDictionary md = new MagicDictionary();
            String[] dict = new String[]{"hello","hallo"};
            md.buildDict(dict);
            System.out.println(md.search("hello"));
        }
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int max = 0;
        int min = 2 << s.length() - 1;
        for(char c :s.toCharArray()){
            map.put(c, map.getOrDefault(c, 0) + 1);
            max = Math.max(max, map.get(c));
        }
        List[] bucket = new List[max + 1];
        for(char c : map.keySet()){
            if(bucket[map.get(c)] == null){
                bucket[map.get(c)] = new ArrayList<Character>();
            }
            bucket[map.get(c)].add(c);
        }
        StringBuilder sb = new StringBuilder();
        for(int i = max; i >= 1; i--){
            if(bucket[i] == null) continue;
            List<Character> list = bucket[i];
            for(char c : list){
                for(int j = 0; j <i; j++){
                    sb.append(c);
                }

            }
        }
        return sb.toString();

    }

    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for(int num : nums) {
            list.add((double) num);

        }

        return  helper(list);


    }

    private boolean helper(List<Double> list){
            if(list.isEmpty()) return false;
            if(list.size() == 1 && list.get(0) - 24.0 <= 0.001){
                return true;
            }
            for(int i = 1; i < list.size(); i++) {
                for(int j = 0; j < i; j++) {
                    Double a = list.get(i);
                    Double b = list.get(j);
                    List<Double> tmp = new ArrayList<>();
                    tmp.addAll(Arrays.asList(a-b, b-a, a*b, a + b));
                    if(Math.abs(b) > 0.001) tmp.add(a/b);
                    if(Math.abs(a) > 0.001) tmp.add(b/a);
                    list.remove(i);
                    list.remove(j);
                    for(Double d : tmp){
                        list.add(d);
                        if(helper(list)) return true;
                        list.remove(d);
                    }
                    list.add(i, a);
                    list.add(j, b);

                 }
            }
            return false;
    }

    class TreeNode{
        TreeNode left;
        TreeNode right;
        int val;
        TreeNode(int val){
            this.val = val;
            this.left = null;
            this.right = null;
        }
    }
    public int widthOfBinaryTree(TreeNode root) {
        if(root == null) return 0;
        Queue<Map.Entry<TreeNode, Integer>> q = new LinkedList<>();
        int max = 0;
        q.offer(new AbstractMap.SimpleEntry<TreeNode, Integer>(root, 1));
        PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> a.compareTo(b));
        while(!q.isEmpty()) {
            int size = q.size();
            int l = q.peek().getValue(), r = 0;
            for(int i = 0; i < size; i++){
                r = q.peek().getValue();
                TreeNode node = q.poll().getKey();
               // if(node.left == null && node.right == null) continue;
                if(node.left != null) q.offer(new AbstractMap.SimpleEntry<TreeNode, Integer>(node.left, 2*r));
                //else q.offer(he);
                if(node.right != null) q.offer(new AbstractMap.SimpleEntry<TreeNode, Integer>(node.right, 2*r + 1));
               // else q.offer(he);
            }
            max = Math.max(max, r - l + 1);
        }
        return max;
    }
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] res = new int[n];
        Deque<Integer> q = new LinkedList<>();
        int pre = 0;
        for(String s : logs){
            String[] array = s.split(":");
            if(array[1].equals("start")){
                if(!q.isEmpty())  res[q.peekFirst()] += Integer.parseInt(array[2]) - pre;
                q.offerFirst(Integer.parseInt(array[0]));
                pre = Integer.parseInt(array[2]);

            }else{
                res[q.peekFirst()] += Integer.parseInt(array[2]) - pre + 1;
                q.pollFirst();
                pre = Integer.parseInt(array[2]) + 1;
            }
        }
        return res;

    }

}
