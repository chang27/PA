package edu.uci.ics.textdb.exp.dictionarymatcher;

/**
 * Created by Chang on 10/10/17.
 */
import java.util.*;
public class Twitter {
    // int tm = 0;
    // class Tweet{
    //     int tweetId;
    //     int timestamp;
    //     Tweet next;
    //     public Tweet(int id){
    //         this.tweetId = id;
    //         this.timestamp = tm++;
    //         next = null;
    //     }
    // }
    // class User{
    //     int userId;
    //     Set<Integer> followees;
    //     Tweet head;
    //     public User(int id){
    //         this.userId = id;
    //         this.followees = new HashSet<>();
    //         followees.add(id);
    //         this.head = null;
    //     }
    //     public void follow(int id){
    //         this.followees.add(id);
    //     }
    //     public void unfollow(int id){
    //         this.followees.remove(id);
    //     }
    //     public void  post(int tweetId){
    //         Tweet tw = new Tweet(tweetId);
    //         tw.next = head;
    //         head = tw;
    //     }
    // }
    // private Map<Integer, User> userMap;
    /**
     * Initialize your data structure here.
     */
    Map<Integer, Set<Integer>> userMap;
    Deque<Integer[]> tq;

    public Twitter() {
        userMap = new HashMap<>();
        tq = new LinkedList<>();
    }

    /**
     * Compose a new tweet.
     */
    public void postTweet(int userId, int tweetId) {
        if (!userMap.containsKey(userId)) {
            userMap.put(userId, new HashSet<>());
        }
        tq.addFirst(new Integer[]{userId, tweetId});
        // if(!userMap.containsKey(userId)){
        //     userMap.put(userId, new User(userId));
        // }
        // userMap.get(userId).post(tweetId);

    }

    /**
     * Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
     */
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if (!userMap.containsKey(userId)) {
            return res;
        }
        Set<Integer> followee = userMap.get(userId);
        Iterator<Integer[]> it = tq.iterator();
        while (it.hasNext() && res.size() < 10) {
            Integer[] next = it.next();
            if (next[0] == userId || followee.contains(next[0])) {
                res.add(next[1]);
            }
        }
        Collections.reverse(res);
        return res;
//         List<Integer> res = new ArrayList<>();
//         if(!userMap.containsKey(userId)){
//             return res;
//         }
//         PriorityQueue<Tweet> q = new PriorityQueue<Tweet>((a, b) -> b.timestamp - a.timestamp);
//         Set<Integer> userSet = userMap.get(userId).followees;
//         for(int u : userSet){
//             Tweet t = userMap.get(u).head;
//             if(t != null){
//                 q.add(t);
//             }

//         }
//         int n = 10;
//         while (!q.isEmpty() && n > 0){
//             Tweet t = q.poll();
//             res.add(t.tweetId);
//             n--;
//             if(t.next != null){
//                 q.add(t.next);
//             }
//         }
//         return res;

    }

    /**
     * Follower follows a followee. If the operation is invalid, it should be a no-op.
     */
    public void follow(int followerId, int followeeId) {
        if (!userMap.containsKey(followerId)) {
            userMap.put(followerId, new HashSet<>());
        }
        userMap.get(followerId).add(followeeId);
        // if(!userMap.containsKey(followerId)){
        //     userMap.put(followerId, new User(followerId));
        // }
        // if(!userMap.containsKey(followeeId)){
        //     userMap.put(followeeId, new User(followeeId));
        // }
        // userMap.get(followerId).follow(followeeId);
    }

    /**
     * Follower unfollows a followee. If the operation is invalid, it should be a no-op.
     */
    public void unfollow(int followerId, int followeeId) {
        //     if(!userMap.containsKey(followerId) || followerId == followeeId){
        //         return;
        //     }
        //     userMap.get(followerId).unfollow(followeeId);
        // }
        if (!userMap.containsKey(followerId) || followerId == followeeId) {
            return;
        }
        userMap.get(followerId).remove(followeeId);
    }

    private static void arrtostring (String s){
        int[] array = new int[26];
        for(int i = 0; i < s.length(); i++) {
            array[s.charAt(i) - 'a']++;
        }
        StringBuilder sb = new StringBuilder();
        for(int j = 0; j < 26 ; j++){
            if(array[j] > 0){
                while(array[j] >0){
                    sb.append((char)(j + 'a'));
                    array[j]--;
                }
            }
        }
        System.out.println(sb.toString());
    }
    List<String> res = new ArrayList<>();
    int max = 0;
    public void removeInvalidParentheses(String s) {
        if(s == null || s.length() == 0) return;
        if(isValid(s)) {
            if(s.length() > max){
                max = s.length();
                res.clear();
                res.add(s);
            }
            else if(s.length() == max){
                res.add(s);
            }
            return;
        }
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) != ')' && s.charAt(i) != '('){
                continue;

            }
            if(i == 0 || s.charAt(i) != s.charAt(i-1)){
                String sub = s.substring(0, i) + s.substring(i+1);
                System.out.println(sub);
                removeInvalidParentheses(sub);
            }


        }
//        int cnt = 0;
//        for(int i = 0; i < s.length(); i++) {
//            if(s.charAt(i) == '(') cnt++;
//            else if (s.charAt(i) == ')') cnt--;
//        }

//        System.out.println(cnt);
//        return dfs(s, 0, cnt);



    }


    public List<String> remove(String s){
       List<String> res = new ArrayList<>();
        if(s == null || s.length() == 0) return null;
        char[] cha = {'(', ')'};
        help(s, 0, 0, res, cha);
        return res;
    }

    private void help(String s, int start, int from, List<String> res, char[] cha){
        System.out.println(s);
        if(s.length() == 0 || s == null) return;

        int cnt = 0;
        for(int j = start; j < s.length(); j++) {
            if(s.charAt(j) == cha[0]) cnt++;
            else if(s.charAt(j) == cha[1]) cnt--;
            if(cnt>=0) continue;
            for(int i = from; i <= j; i++){
                if(s.charAt(i) == cha[1] && (i == 0 || s.charAt(i) != s.charAt(i-1))){
                    help(s.substring(0, i) + s.substring(i+1), j, i, res, cha );
                }
            }
            return;
        }

        if(isValid(s)){
            res.add(s);
        }
    //    StringBuilder sb = new StringBuilder(s).reverse();
    //    if(cha[0] == '('){
    //        help(sb.toString(), 0,0, res, new char[]{')', '('});

     //   }else{
           // res.add(sb.toString());
      //      the = sb.toString();
      //  }

    }

    private boolean isValid(String s) {
        if(s == null) return false;
        int cnt = 0;
        for (int i = 0; i < s.length(); i++) {

            if (s.charAt(i) == '(') {
                cnt++;
            } else if (s.charAt(i) == ')') {
                if (cnt == 0) return false;
                cnt--;
            }
        }
        return cnt == 0;
    }

    public int largestPalindrome(int n) {
        if(n == 1) return 9;
        long maxValue = (long) Math.pow(10, n) - 1; //99
        long minValue = (long) Math.pow(10, n-1);//10
        long maxP = maxValue * maxValue;//9801
        long start = maxP/ (long) Math.pow(10, n);//98
        while(true){
            long res = getP(start--);
            if(res > maxP) continue;
            for(long i = maxValue; i >= minValue; i--){
                if(res / i > maxValue){
                    break;
                }
                if(res % i == 0) return (int) (res % 1337);
            }


        }

    }

    private long getP(long num){
       // String the = "abc";
        String s = String.valueOf(num) + new StringBuilder().append(num).reverse().toString();
        return Long.parseLong(s);
    }

    public String splitLoopedString(String[] strs) {
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < strs.length; i++){
            String rev = new StringBuilder(strs[i]).reverse().toString();
            strs[i] = strs[i].compareTo(rev) >= 0 ? strs[i] : rev;
            sb.append(strs[i]);
        }
        String result = sb.toString();
        sb.delete(0, sb.length());
        int cur = 0;
        for(int i = 0; i < strs.length; i++) {
            String mid = result.substring(cur + strs[i].length()) + result.substring(0, cur);
            String t1 = strs[i];
            String t2 = new StringBuilder(strs[i]).reverse().toString();
            for(int j = 0; j < t1.length(); j++){
                String s1 = t1.substring(j) + mid + t1.substring(0, j);
                String s2 = t2.substring(j) + mid + t2.substring(0, j);
                if(s1.compareTo(result) > 0) result = s1;
                if(s2.compareTo(result) > 0) result = s2;
            }
            cur += strs[i].length();
            Deque<Integer> dq = new LinkedList<>();
            dq.push(1);
            dq.pop();
            Map<Integer, Integer> map1 = new HashMap<>();
            map1.put(1, 2);
            Map<Integer, Integer> map2 = new HashMap<>(map1);

        }
        return result;




    }

    public String getLongestString(String[] strs, String s) {
        int n = s.length();
        Map<Integer, List<String>> map = new HashMap<>();
        for(String str : strs){
            int len = str.length();
            if(! map.containsKey(len)){
                map.put(len, new ArrayList<>());
            }
            map.get(len).add(str);
        }
        Queue<String> q = new LinkedList<>();
        q.offer(s);
        String candi = s;
        while(! q.isEmpty()){
            candi = q.poll();
            List<String> next = getNext(candi, map);
            if(next.isEmpty()){
                continue;
            }
            q.addAll(next);

        }
        return candi;
    }

    private List<String> getNext(String s, Map<Integer, List<String>> map){
        List<String> res = new ArrayList<>();
        int len = s.length();
        if(! map.containsKey(len + 1)){
            return res;
        }

        for(String ss : map.get(len +1 )){
            boolean flag = false;
            for(char c : s.toCharArray()){
                if(ss.indexOf(c) == -1){
                    flag = true;
                    break;
                }
            }
           if(!flag) res.add(ss);
        }
        return res;
    }

    public List<String> wordsAbbreviation(List<String> dict) {
        int n = dict.size();
        int[] prefix = new int[n];
        Arrays.fill(prefix, 1);
        List<String> res = new ArrayList<>();
        List<Integer> todo = new ArrayList<>();
        for(int i = 0; i < n; i++){
            todo.add(i);
        }
        Map<String, List<Integer>> map = new HashMap<>();
        while(! todo.isEmpty()) {
            for(int pos : todo) {
                String key = getAbbr(dict.get(pos), prefix[pos]);
                if (!map.containsKey(key)) {
                    map.put(key, new ArrayList<>());
                }
                map.get(key).add(pos);

            }

                todo.clear();
            Set<String> set = map.keySet();
                for(String k : set){
                    if(map.get(k).size() > 1){
                        todo.addAll(map.get(k));
                    }
                    map.remove(k);
                }
            }

//        for(int s : map.values().){
//            res.add(s);
//        }
        return res;

    }

    private String getAbbr(String s, int from) {
        if(from >= s.length() - 2) return s;

        StringBuilder sb = new StringBuilder();
        sb.append(s.substring(0, from));
        sb.append(s.length() - 1 - from);
        sb.append(s.substring(s.length() - 1));
        return sb.toString();
    }

    public static void main(String[] args){
//        String s = "((a)()()";
//        Twitter tw = new Twitter();
//      List<String> asdf = tw.remove(s);
//        System.out.println("result: " + asdf.toString());
//        String ss ="        file.txt\\";
//        int cnt = 0;
//        int offset = 0;
//        while(ss.length() > 0){
//            if(ss.startsWith("\t", offset)){
//                cnt++;
//                offset += 1;
//            }else break;
//        }
//        long num = 1;
//        String str = String.valueOf(num);
//        System.out.println(ss.lastIndexOf("\t"));
//        System.out.println(cnt);
        String[] strs = {"i", "sim", "im", "asim", "bcdew"};
        String start = "i";
        Twitter tw = new Twitter();
        String res = tw.getLongestString(strs, start);
        System.out.println(res);

    }
}

