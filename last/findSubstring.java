package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/22/17.
 */
public class findSubstring {

    public List<Integer> findSubstring(String s, List<String> words) {
        List<Integer> res = new ArrayList<>();
        if (s.length() == 0 || words.size() == 0 || s.length() < words.size() * words.get(0).length()) {
            return res;
        }
        int n = s.length();
        int cnt = words.size();
        int len = words.get(0).length();
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        for (int i = 0; i < len; i++) {
            int left = i, c = 0;
            Map<String, Integer> map2 = new HashMap<>();
            for (int j = i; j + len <= n; j += len) {
                String str = s.substring(j, j + len);
                if (map.containsKey(str)) {
                    map2.put(str, map2.getOrDefault(str, 0) + 1);

                    if (map2.get(str) <= map.get(str)) c++;
                    while (map2.get(str) > map.get(str)) {
                        String tmp = s.substring(left, left + len);
                        map2.put(tmp, map2.get(tmp) - 1);
                        if (map2.get(tmp) < map.get(tmp)) c--;
                        left += len;
                    }
                    if (c == cnt) {
                        res.add(left);
                        String sub = s.substring(left, left + len);
                        map2.put(sub, map2.get(sub) - 1);
                        left += len;
                        c--;
                    }
                } else {
                    map2.clear();
                    left = j+ len;
                    c = 0;
                }
            }
        }
        return res;
    }
    public List<Integer> findSubstring1(String s, List<String> words) {
        List<Integer> res = new ArrayList<>();
        int n = s.length();
        int m = words.size();
        int len = words.get(0).length();
        Map<String, Integer> map = new HashMap<>();
        for(String ss : words){
            map.put(ss, map.getOrDefault(ss, 0) + 1);
        }
        for(int i = 0; i + m*len <= n; i++){
            Map<String, Integer> map2 = new HashMap<>();
            int cnt = 0, j = i;
            for(;j <= i + (m-1) * len; j += len){
                String str = s.substring(j, j + len);

                if(!map.containsKey(str)) break;
                map2.put(str, map2.getOrDefault(str, 0 )+ 1);
                if(map2.get(str) > map.get(str)) break;

            }
            if(j == i + m*len) res.add(i);
        }
        return res;
    }

    public static List<String> split (String s){
        String[] arr = s.split(",");
        System.out.println(arr.length);
        List<String> res = new ArrayList<>(Arrays.asList(arr));
        return res;
    }


    public String nearestPalindromic(String n) {
        if(n == null || n.length() <= 1) return n;
        char[] arr = n.toCharArray();
        for(int i = 0; i < n.length()/2 ; i++){
            arr[n.length() -i - 1] = arr[i];
        }
        String b = arr.toString();
        if(Long.parseLong(arr.toString()) >= Long.parseLong(n)) {

            for(int i = (n.length() -1)/ 2; i >= 0; i--) {
                if(--arr[i] <'0'){
                    arr[i] = '9';
                }else break;
            }
            if(arr[0] == '0'){
                Arrays.fill(arr, '9');

            }
        }else{
            for(int i = (n.length() - 1)/2; i >= 0; i--){
                if(++arr[i] > '9'){
                    arr[i] = '0';
                }else{
                    break;
                }
            }
            for(int i = 0; i < n.length()/2 ; i++){
                arr[n.length() -i - 1] = arr[i];
            }

        }
        String t = String.valueOf(arr);
        if(Math.abs(Long.parseLong(n)- Long.parseLong(b)) <= Math.abs(Long.parseLong(n) - Long.parseLong(t))){
            return b;
        }
        return t;

    }
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        if(words == null || words.length == 0) res.add("");

        for (int i = 0, w; i < words.length; i = w) {
            int len = -1;
            for(w = i; w < words.length && len + words[w].length() + 1 <= maxWidth; w++){
                len += words[w].length() + 1;
            }
            int num = w - i;
            int even = 1;
            int more = 0;
            StringBuilder sb = new StringBuilder();
            if(num > 1 && w != words.length){
                even += (maxWidth - len)/(num - 1);
                more = (maxWidth - len)%(num - 1);
            }
            for(int j = i; j < w; j++){
                sb.append(words[j]);
                for(int k = 0; k < even; k++){
                    sb.append(' ');
                }
                if(more > 0){
                    sb.append(' ');
                    more--;
                }
            }
            while(maxWidth > sb.length()){
                sb.append(' ');
            }
            res.add(sb.toString());
        }
        return res;

    }

    public List<String> fullJustify1(String[] words, int maxWidth) {
        List<String> list = new LinkedList<String>();

        for (int i = 0, w; i < words.length; i = w) {
            int len = -1;
            for (w = i; w < words.length && len + words[w].length() + 1 <= maxWidth; w++) {
                len += words[w].length() + 1;
            }
            int num = w - i;
            int even = 1;
            int more = 0;
            StringBuilder sb = new StringBuilder();
            if(num > 1 && w != words.length){
                even += (maxWidth - len)/(num - 1);
                more = (maxWidth - len)%(num - 1);
            }
            for(int j = i; j < w; j++){
                sb.append(words[j]);
                for(int k = 0; k < even; k++){
                    sb.append(' ');
                }
                if(more > 0){
                    sb.append(' ');
                    more--;
                }
            }
            while(maxWidth > sb.length()){
                sb.append(' ');
            }
            list.add(sb.toString());

//            StringBuilder strBuilder = new StringBuilder(words[i]);
//            int space = 1, extra = 0;
//            if (w != i + 1 && w != words.length) { // not 1 char, not last line
//                space = (L - len) / (w - i - 1) + 1;
//                extra = (L - len) % (w - i - 1);
//            }
//            for (int j = i + 1; j < w; j++) {
//                for (int s = space; s > 0; s--) strBuilder.append(' ');
//                if (extra-- > 0) strBuilder.append(' ');
//                strBuilder.append(words[j]);
//            }
//            int strLen = L - strBuilder.length();
//            while (strLen-- > 0) strBuilder.append(' ');
//            list.add(strBuilder.toString());
        }

        return list;
    }

    public static String minNext(String input) {
        Map<Integer, Character> map = new HashMap<>();
        map.put(0, '2');
        map.put(1, '3');
        map.put(3, '5');
        map.put(4, '9');
        TreeSet<Character> list = new TreeSet<>();
        for(int i = 0; i < input.length(); i++){
            if(i != 2){
                list.add(input.charAt(i));
            }
        }
        System.out.println(list.toString());
        StringBuilder sb = new StringBuilder();
        for(int i = 4; i >= 0; i--){
            if(i == 2) {
                sb.append(':');
                continue;
            }
            char num =  input.charAt(i);
            if(list.higher(num) != null && list.higher(num) <= map.get(i)){

                sb.append(list.higher(num));
                break;
            }else if(list.higher(num) != null && i == 1 && input.charAt(i-1) <= '1'){
                sb.append(list.higher(num));

            }
            else{
                sb.append(list.first());
            }
        }
        if(sb.length() < 5){
            for(int i = 5 - sb.length()- 1; i >= 0; i--){
                sb.append(input.charAt(i));
            }
        }
        return sb.reverse().toString();
    }

//    public static void main(String[] args){
//        String input = "23:59";
//        String out = minNext(input);
//        System.out.println(out);
//        List<String> list = new ArrayList<>();
//        list.add("a");
//        list.add("ab");
//        list.add("abc");
//        list.add("tabc");
//        list.add("tabqc");
//        list.add("z");
//        String result = longestWord(list);
//        System.out.println(result);
//
//    }

    public static String longestWord(List<String> words) {
        Map<Integer, List<String>> map = new HashMap<>();
        for(String w : words){
            if(!map.containsKey(w.length())){
                map.put(w.length(), new ArrayList<>());
            }
            map.get(w.length()).add(w);
        }
        String max = "";
        if(words.get(1).isEmpty()) return null;
        for(int i = 0; i < map.get(1).size(); i++){
            String str = helper(map.get(1).get(i), 2, map);
            if(max.length() < str.length()){
                max = str;
            }
        }
        return max;
    }
    private static String helper(String s, int index, Map<Integer, List<String>> map) {
        if(!map.containsKey(index)){
            return s;
        }
        String max = s;
        for(int i = 0; i < map.get(index).size(); i++){
            if(valid(s, map.get(index).get(i))){
                String next = helper(map.get(index).get(i), index + 1, map);
                if(max.length() < next.length()){
                    max = next;
                }
            }

        }
        return max;
    }
    private static boolean valid(String a, String b){
        for(int i = 0; i < a.length(); i++){
            if(b.indexOf(a.charAt(i)) == -1) return false;
        }
        return true;
    }

    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] root = new int[n];
        for(int i = 0; i < n; i++){
            root[i] = i;
        }
        for(int[] edge : edges){
            int root1 = find(root, edge[0]);
            int root2 = find(root, edge[1]);
            if(root1 == root2) return edge;
            root[root1] = root2;
        }
        return new int[2];

    }

    /**
     * This is difficult!!!
     * @param edges
     * @return
     */
    public static int[] findRedundantDirectedConnection(int[][] edges) {
        int[] can1 = {-1, -1};
        int[] can2 = {-1, -1};
        int[] parent = new int[edges.length + 1];
        for (int i = 0; i < edges.length; i++) {
            if (parent[edges[i][1]] == 0) {
                parent[edges[i][1]] = edges[i][0];
            } else {
                can2 = new int[] {edges[i][0], edges[i][1]};
                can1 = new int[] {parent[edges[i][1]], edges[i][1]};
                edges[i][1] = 0;
            }
        }
//        for (int i = 0; i < edges.length; i++) {
//            parent[i] = i;
//        }
        for (int i = 0; i < edges.length; i++) {
            if (edges[i][1] == 0) {
                continue;
            }
            int child = edges[i][1], father = edges[i][0];
            if (find(parent, father) == child) {
                if (can1[0] == -1) {
                    return edges[i];
                }
                return can1;
            }
         //   parent[child] = father;
        } return can2;
    }

    private static int find(int[] root, int i){
        while(i != root[i]){
            root[i] = root[root[i]];
            i = root[i];
        }
        return i;
    }

    public int integerReplacement(int n) {
        int cnt = 0;
        while(n != 1){
            if(n % 2 == 0){
                n /= 2;
            }else{
                if(n == 3 || ((n >> 1) & 1) == 0) {
                    n--;
                }else{
                    n++;
                }
            }
            cnt++;
        }
        return cnt;
    }

    public static List<Integer> generateSub(int k){
        List<Integer> res = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        cur.add(0);
        for(int i = 0; i < k; i++){
            List<Integer> tmp = new ArrayList<>();
            for(int c : cur){
            for(int j = k-1; j >=0 ; j--){
                    if(( c >> j) != 0) break;
                    //if((1 << j ^ c) != 0) break;
                    System.out.println(1 << j | c);
                    tmp.add(1 << j | c);
                }
            }
            res.addAll(cur);
            cur = tmp;
        }
        res.addAll(cur);
        return res;
    }


    private String getKey(String s){
        char[] array = new char[3];
        array[0] = s.charAt(0);
        array[1] = (char) (s.length() - 2);
        array[2] = s.charAt(s.length() - 1);
        return new String(array);
    }

    public static void main(String[] args){
//        int[][] edges = {{4,2}, {1, 5}, {5, 2}, {5,3}, {2, 4}};
//        int[][] edges1 = {{1, 2}, {1, 3}, {2, 3}};
//        int[] res = findRedundantDirectedConnection(edges1);
//        System.out.println(res[0] + " " + res[1]);

//        ArrayList<Integer> a = new ArrayList<>();
//        a.add(1);
//        a.add(2);
//        ArrayList<Integer> b = new ArrayList<>(a);
//        ArrayList<Integer> c = a;
//        a.add(3);
//     List<Integer> res = generateSub(3);
//        System.out.println(res.toString());
        int i = 5;
        while(i-- > 0){
            System.out.println(i);
        }

   //     System.out.println(c.toString());
//        String output = minNext("13:55");
//        System.out.println(output);
    }

}
