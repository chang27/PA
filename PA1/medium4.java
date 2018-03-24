public int canCompleteCircuit(int[] gas, int[] cost) {
        int start = 0;
        int tank = 0;
        int total = 0;
        for(int i = 0; i < gas.length; i++){
            tank += gas[i] - cost[i];
            total += tank;
            if(tank < 0){
                start = i+1;
                tank = 0;
            }
        }
        if(total < 0) return -1;
        return start;
        
        
    }

    public String removeKdigits(String num, int k){
        int digits = num.length() - k;
        Stack<Character> stack = new Stack<>();
        for(int i = 0; i < num.length(); i++){
            while(!stack.isEmpty() && stack.top() > num.charAt(i) && k > 0){
                stack.pop();
                k--;
            }

            stack.push(num.charAt(i));
        }
        StringBuilder sb = new StringBuilder();
        while(!stack.isEmpty()){
            sb.append(stack.pop());
        }
        sb.reverse();
        while(sb.length() > 1 && sb.charAt(0) == '0') sb.deleteCharAt(0);
        return sb.toString();
    }


    public String removeKdigits(String num, int k) {
        int digit = num.length() - k;
        if(num == null || num.isEmpty() || k ==0) return num;
        if(num.length() <= k) return "0";
        
        char[] newChar = new char[num.length()];
        int top = 0;
        for(int i = 0; i < num.length(); i++){
            while( top > 0 && newChar[top-1] > num.charAt(i) && k > 0){
                top--;
                k--;
            }

            newChar[top++] = num.charAt(i); 
        }
        int index = 0;
        while(index < digit && newChar[index] == '0'){
            index++;
        }
        return index == digit? "0" : new String(newChar, index, digit - index);
        
        
        
    }

    class Trienode{
    public boolean isWord;
    public Trienode[] children = new Trienode[26];
    public Trienode(){};
}

public class Trie {
    private Trienode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new Trienode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Trienode ws = root;
        for(int i = 0; i < word.length(); i++){
            if(ws.children[word.charAt(i) - 'a'] == null){
                ws.children[word.charAt(i) - 'a'] = new Trienode();
            }
                ws = ws.children[word.charAt(i) - 'a'];
            
        }
        ws.isWord = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        Trienode point = root;
        for(int i = 0; i < word.length(); i++){
            if (point.children[word.charAt(i) - 'a'] == null) return false;
            point = point.children[word.charAt(i) - 'a'];
            
        }
        return point.isWord;
        
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        Trienode pointer = root;
        for(int i = 0; i < prefix.length(); i++){
            if(pointer.children[prefix.charAt(i) - 'a'] == null) return false;
            pointer = pointer.children[prefix.charAt(i) - 'a'];
        }
        return true;
        
    }
    public String getMP(String input){
        StringBuilder sb = new StringBuilder();
        TrieNode p = root;
        int preMatch = 0;
        for(int i =0; i < input.length(); i++){
            if(p.children[input.charAt(i) - 'a'] != null){
                p = p.children[input.charAt(i) - 'a'];
                if(p.isEnd){
                    preMatch = i+1;
                }

            }else break;
        }
        return input.substring(0, preMatch);
    }

}

public class WordDictionary {
    class TrieNode{
        public boolean isWord;
        public TrieNode[] children = new TrieNode[26];
        public TrieNode(){};
    }
    private TrieNode root;
    /** Initialize your data structure here. */
    public WordDictionary() {
        root = new TrieNode();  
    }
    
    /** Adds a word into the data structure. */
    public void addWord(String word) {
        TrieNode n = root;
        for(int i = 0; i < word.length(); i++){
            if(n.children[word.charAt(i) - 'a'] == null) {
                n.children[word.charAt(i) - 'a'] = new TrieNode();
            } 
            n = n.children[word.charAt(i) - 'a'];
        }
        n.isWord = true;
    }
    
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
        return match(word.toCharArray(), 0, root);       
    }
    
    private boolean match(char[] chs, int k, TrieNode node){
        if(k == chs.length) return node.isWord;
        if(chs[k] == '.'){
            for(int i = 0; i < node.children.length; i++){
                if(node.children[i] != null && match(chs, k+1, node.children[i])) return true;
            }
        }else{
            return node.children[chs[k] - 'a'] != null && match(chs, k+1, node.children[chs[k] - 'a']);
        }
        
       return false; // this false is for (if condition) when no matches found, then a final return false.
        
    }
}
//These characteristics suggest[further explanation needed] a simple and fast method of translating a binary value into the corresponding Gray code. 
//Each bit is inverted if the next higher bit of the input value is set to one. This can be performed in parallel by a bit-shift and exclusive-or operation if they are available: 
//the nth Gray code is obtained by computing {\displaystyle n\oplus \lfloor n/2\rfloor } n\oplus \lfloor n/2\rfloor 

A similar method can be used to perform the reverse translation, but the computation of each bit depends on the computed value of the next higher bit so it cannot be performed in parallel. Assuming {\displaystyle g_{i}} g_{i} is the {\displaystyle i} ith gray-coded bit ( {\displaystyle g_{0}} g_{0} being the most significant bit), and {\displaystyle b_{i}} b_{i} is the {\displaystyle i} ith binary-coded bit ( {\displaystyle b_{0}} b_{0} being the most-significant bit), the reverse translation can be given recursively: {\displaystyle b_{0}=g_{0}} b_{0}=g_{0}, and {\displaystyle b_{i}=g_{i}\oplus b_{i-1}} b_{i}=g_{i}\oplus b_{i-1}. Alternatively, decoding a Gray code into a binary number can be described as a prefix sum of the bits in the Gray code, where each individual summation operation in the prefix sum is performed modulo two.

To construct the binary-reflected Gray code iteratively, at step 0 start with the {\displaystyle code_{0}=0} code_{0}=0, and at step {\displaystyle i>0} i>0 find the bit position of the least significant 1 in the binary representation of {\displaystyle i} i and flip the bit at that position in the previous code {\displaystyle \mathrm {code} _{i-1}} \mathrm {code} _{i-1} to get the next code {\displaystyle code_{i}} code_{i}. The bit positions start 0, 1, 0, 2, 0, 1, 0, 3, ... (sequence A007814 in the OEIS). See find first set for efficient algorithms to compute these values.

public List<Integer> grayCode(int n){
    List<Integer> result = new LinkedList<>();
    for(int i = 0; i < 1<<n; i++){
        result.add( i ^ i >> 1);
    }
    return result;
}

public List<Integer> grayCode(int n){
    List<Integer> result = new ArrayList<>();
    result.add(0);
    for(int i = 0; i < n; i++){
        for(int j = result.size() - 1; j>=0; j--){
            result.add( 1<<i|result.get(j));
        }

    }
    return result;
}




public int calculate(String s){
    int len = s.length();
    if(s == null || len == 0|| s.trim().length == 0) return 0;
    int num = 0;
    int res = 0;
    char sign =  '+';
    Stack<Integer> stack = new Stack<>();
    for(int i = 0; i < len; i++){
        if(s.charAt(i) >= '0' && s.charAt(i) <='9'){
            num = num *10 + s.charAt(i) - '0';
        }
        if(s.charAt(i) == '+' || s.charAt(i) == '-' || s.charAt(i) == '*' || s.charAt(i) == '/' || i = len -1){
            if(sign == '+' || sign == '-'){
                int tmp = sign=='+'? num: -num;
                res += tmp;
                stack.push(tmp);
            }
            else{
                res -= stack.peek();
                int tmp = sign == '*' ? stack.pop()*num : stack.pop()/num;
                stack.push(tmp);
                res += tmp;
            }
            sign = s.charAt(i);
            num = 0;
        }
    }
    return res;
}


 public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    if(n == 1) return Collections.singletonList(0);
    if(n == 2) return Arrays.asList(0, 1);
    List<Set<Integer>> adj = new ArrayList<>();
    for(int i = 0; i < n; i++) adj.add(new HashSet<>());
    for(int[] edge : edges){
        adj.get(edge[0]).add(edge[1]);
        adj.get(edge[1]).add(edge[0]);
    }
    LinkedList<Integer> leaves = new LinkedList<>();// Want to use pollFirst() must define linkedlist.
    for(int i = 0; i < n; i++){
        if(adj.get(i).size() == 1) leaves.add(i);
    }
    while(n > 2){ 
        int size = leaves.size();
        n -= size;
        for(int i = 0; i < size; i++){
            int m = leaves.pollFirst();
            int k = adj.get(m).iterator().next();
            adj.get(k).remove(m);
            if(adj.get(k).size() == 1) leaves.addLast(k);
        }
    }
    return leaves;
        
    }


     public boolean canFinish(int numCourses, int[][] prerequisites) {
        ArrayList<Set<Integer>> adj = new ArrayList<>(numCourses);
        for(int i = 0; i < numCourses; i++){
            adj.add(new HashSet<>()); 
        }
        for(int[] pre :prerequisites){
            adj.get(pre[1]).add(pre[0]);
        }
        boolean[] visited = new boolean[numCourses];
        boolean[] finished = new boolean[numCourses];
        for(int i = 0; i < numCourses; i++){
            if(dfsCycle(adj, i, visited, finished))
                return false;
        }
        return true;
        
    }
    private boolean dfsCycle(ArrayList<Set<Integer>> graph, int i, boolean[] visited, boolean[] finished){
        if(finished[i] ) return false;
        if (visited[i]){ return true;}
        else{
            visited[i] = true;
        }
        for(int n : graph.get(i)){
            if (dfsCycle(graph, n, visited, finished))
                return true;
        }
        finished[i] = true;
        return false;

    }

    public boolean canFinish(int numCourses, int[][] prerequisites){
        ArrayList<Set<Integer>> adj = new ArrayList<>(numCourses);
        for(int i = 0; i < numCourses; i++){
            adj.add(new HashSet<Integer>());
        }
        int[] degree = new int[numCourses];
        for(int[] pre : prerequisites){
            adj.get(pre[1].add(pre[0]));
            degree[pre[0]]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for(int i = 0; i < numCourses; i++){
            if (degree[i] == 0) {q.offer(i);}
        }
        int cnt = 0;
        while(!q.isEmpty()){
            int s = q.pop();
            for(int a : adj.get(s)){
                degree[a]--;
                cnt++;
                if(degree[a] == 0) q.offer(a);
            }
        }
        return cnt == prerequisites.length;

    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        ArrayList<Set<Integer>> adj = new ArrayList<>();
        for(int i = 0; i < numCourses; i++){
            adj.add(new HashSet<>());
        }
        int[] degree = new int[numCourses];
        for(int[] pre: prerequisites){
            adj.get(pre[1]).add(pre[0]);
            degree[pre[0]]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for(int i = 0; i < numCourses; i++){
            if(degree[i] == 0) q.offer(i);
        }
        ArrayList<Integer> res = new ArrayList<>();
        while(!q.isEmpty()){
            int n = q.poll();
            res.add(n);
            for(int neighbor : adj.get(n)){
                degree[neighbor]--;
                if(degree[neighbor] == 0) q.offer(neighbor);
            }
        }
        if (res.size() == numCourses) {
            return res.stream().mapToInt(i -> i).toArray();
        }
        else return new int[0];               
    }


int[][] dp;
public NumMatrix(int[][] matrix) {
    if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return;
    int m = matrix.length;
    int n = matrix[0].length;
    dp = new int[m + 1][n + 1];
    for(int i = 1; i <= m; i++) {
        for (int j = 1; i <= n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
        }
    }
}
public int sumRegion(int row1, int col1, int row2, int col2){
    return dp[row2+1][col2+1]- dp[row1][col2+1] - dp[row2+1][col1] + dp[row1][col1];
}

public String largestNumber(int[] num){
    if(num == null || num.length = 0){
        return "";
    }
    String[] numToString = new String[num.length];
    for(int i = 0; i < num.length; i++){
        numToString[i] = String.valueOf(num[i]);
    }
    Comparator<String> comp = new Comparator<String>(){
        @Override
        public int compare(String str1, String str2){
            String s1 = str1 + str2;
            String s2 = str2 + str1;
            return s2.compareTo(s1);
        }
    }
    Arrays.sort(numToString, comp);
    if(numToString[0].charAt(0) == '0'){
        return "0";
    }
    StringBuilder sb = new StringBuilder();
    for(String s: numToString){
        sb.append(s);
    }
    return sb.toString();

}

public boolean find132pattern(int[] nums) {
        if(nums == null || nums.length <= 2){
            return false;
        }
        int second = Integer.MIN_VALUE;
        Stack<Integer> stack = new Stack<>();
        for(int i = nums.length -1; i >= 0; i--){
            if(nums[i] < second) return true;
            while(! stack.isEmpty() && nums[i] > stack.peek()){
                second = stack.pop();
            }
            stack.push(nums[i]);
        }
        return false;
        
    }


public boolean checkSubarraySum(int[] nums, int k){
    if(nums == null || nums.length <= 1) return false;
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, -1);
    int preSum = 0;
    for(int i = 0; i < nums.length; i++){
        preSum += nums[i];
        if(k != 0) preSum %= k;
        if(map.containsKey(preSum)){
            if(i - map.get(preSum) > 1) return ture;
        }else map.put(preSum, i); // here if preSum exists, but length less than 2: not update the arrayindex.
        
    }
    return false;
}

public int subarraySum(int[] nums, int k) {
        if(nums == null) return 0;
        int result = 0;
        int sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);// this is for sum = k, make sure (sum-k == 0) exists.
        for(int i = 0; i < nums.length; i++){
            sum += nums[i];
            if(map.containsKey(sum - k)){
                result += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return result;
        
    }


    public int minimumTotal(List<List<Integer>> triangle) {
        int m = triangle.size();
        int n = triangle.get(m-1).size();
        int[] dp = new int[n];
        for(int i = m-1; i >= 0; i--){
            for(int j = 0; j < triangle.get(i).size(); j++) {
                if(i == m - 1){
                    dp[j] = triangle.get(i).get(j);
                }else{
                   dp[j] = Math.min(dp[j], dp[j+1]) + triangle.get(i).get(j); 
                }
                
                
            }
        }
        return dp[0];
        
    }

    public int leastInterval(char[] tasks, int n){
        int[] c = new int[26];
        int max = 0, numMax = 0;
        for(char t: tasks){
            c[t - 'A']++;
            if(c[t - 'A'] > max){
                max = c[t - 'A'];
                numMax = 1;
            }else if(c[t-'A'] == max){
                numMax++;
            }
            
        }
        return Math.max(tasks.length, (max -1) * (n+1) + numMax);

    }
public List<Integer> spiralOrder(int[][] matrix){
    List<Integer> res = new ArrayList<>();
    if(matrix.length == 0) return res;
    int rowB = 0, rowE = matrix.length -1, colB = 0, colE = matrix[0].length - 1;
    while(rowB <= rowE && colB <= colE){
        for(int i = colB; i < colE; i++){
            res.add(matrix[rowB][i]);
        }
        rowB++;
        for(int j = rowB; j <= rowE; j++){
            res.add(matrix[j][colE]);
        }
        colE--;
        if(rowB <= rowE){
            for(int i = colE; i >= colB; i--){
                res.add(matrix[rowE][i]);
            }
            rowE--;
        }
        
        if(colB <= colE){
            for(int j = rowE; j >= rowB; j--){
                res.add(matrix[j][colB]);
            }
            colB++;

        }
    }
    return res;
}
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int rowB = 0, rowE = n-1;
        int colB = 0, colE = n-1;
        int e = 1;
        while(rowB <= rowE && colB <= colE){
            for(int i = colB; i <= colE; i++){
                matrix[rowB][i] = e++;
            }
            rowB++;
            for(int i = rowB; i <= rowE; i++){
                matrix[i][colE] = e++;
            }
            colE--;
            
            if(rowB > rowE) break;
            for(int i = colE; i >= colB; i--){
                matrix[rowE][i] = e;
                e += 1;
            }
            rowE--;
            
            if(colB > colE) break;
            for(int i = rowE; i >= rowB; i--){
                matrix[i][colB] = e++;
            }
            colB++;
        }
       return  matrix;
    }

    public class Twitter {
        private static int timeStamp = 0;
        private Map<Integer, User> userMap;
        private class Tweet{
            public int id;
            public int time;
            public Tweet next;
            public Tweet(int id){
                this.id = id;
                this.time = timeStamp++;
                this.next = null;
            }
        }
        public class User{
            public int id;
            public Set<Integer> followed;
            public Tweet tweet-head;
            public User(int id){
                this.id = id;
                followed = new HashSet<>();
                follow(id);
                tweet-head = null;
            }
            public void follow(int id){
                followed.ad(id);
            }
            public void unfollow(int id){
                followed.remove(id);
            }
            public void post(int id){
                Tweet t = new Tweet(id);
                t.next = tweet-head;
                tweet-head = t;
            }
        }
        public Twitter(){
            userMap = new HashMap<Integer, User>();
        }
        public void postTweet(int userId, int tweetId){
            if(!userMap.contains(userId)){
                userMap.put(userId, new User(userId));
            }
            usrMap.get(userId).post(tweetId);
        }

        public List<Integer> getNewsFeed(int userId){
            List<Integer> res = new LinkedList<>();
            if(!userMap.containsKey(userId)) return res;
            Set<Integer> users = userMap.get(userId).followed;
            PriorityQueue<Tweet> q = new PriorityQueue<Tweet>(users.size(), (a,b)->(b.time - a.time));
            for(int user : users){
                Tweet t = userMap.get(user).tweet-head;
                if(t != null){ q.add(t);}
            }
            int n = 0;
            while(!q.isEmpty() && n < 10){
                Tweet t = q.poll();
                res.add(t.id);
                n++;
                if(t.next != null){
                    q.add(t.next);
                }
            }
            return res;
        }
        public void follow(int followerId, int followeeId){
            if(!userMap.containsKey(followerId)){
                userMap.put(followerId, new User(followerId));
            }
            if(!userMap.containsKey(followeeId)){
                userMap.put(followeeId, new User(followeeId));
            }
            userMap.get(followerId).follow(followeeId);
        }

        public void unfollow(int followerId, int followeeId){
            if(!userMap.containsKey(followerId) || followerId == followeeId){
                return;
            }
            userMap.get(followerId).unfollow(followeeId);
        }


public class Twitter {
    int tm = 0;
    class Tweet{
        int tweetId;
        int timestamp;
        Tweet next;
        public Tweet(int id){
            this.tweetId = id;
            this.timestamp = tm++;
            next = null;
        }
    }
    class User{
        int userId;
        Set<Integer> followees;
        Tweet head;
        public User(int id){
            this.userId = id;
            this.followees = new HashSet<>();
            followees.add(id);
            this.head = null;
        }
        public void follow(int id){
            this.followees.add(id);
        }
        public void unfollow(int id){
            this.followees.remove(id);
        }
        public void  post(int tweetId){
            Tweet tw = new Tweet(tweetId);
            tw.next = head; 
            head = tw;
        }
    }
    private Map<Integer, User> userMap;
    /** Initialize your data structure here. */
    public Twitter() {
        userMap = new HashMap<>();    
    }
    
    /** Compose a new tweet. */
    public void postTweet(int userId, int tweetId) {
        if(!userMap.containsKey(userId)){
            userMap.put(userId, new User(userId));
        }
        userMap.get(userId).post(tweetId);
        
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if(!userMap.containsKey(userId)){
            return res;
        }
        PriorityQueue<Tweet> q = new PriorityQueue<Tweet>((a, b) -> b.timestamp - a.timestamp);
        Set<Integer> userSet = userMap.get(userId).followees;
        for(int u : userSet){
            Tweet t = userMap.get(u).head;
            if(t != null){
                q.add(t);
            }
            
        }
        int n = 10;
        while (!q.isEmpty() && n > 0){
            Tweet t = q.poll();
            res.add(t.tweetId);
            n--;
            if(t.next != null){
                q.add(t.next);
            }
        }
        return res;
        
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    public void follow(int followerId, int followeeId) {
        if(!userMap.containsKey(followerId)){
            userMap.put(followerId, new User(followerId));
        }
        if(!userMap.containsKey(followeeId)){
            userMap.put(followeeId, new User(followeeId));
        }
        userMap.get(followerId).follow(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    public void unfollow(int followerId, int followeeId) {
        if(!userMap.containsKey(followerId) || followerId == followeeId){
            return;
        }
        userMap.get(followerId).unfollow(followeeId);
    }
}

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter obj = new Twitter();
 * obj.postTweet(userId,tweetId);
 * List<Integer> param_2 = obj.getNewsFeed(userId);
 * obj.follow(followerId,followeeId);
 * obj.unfollow(followerId,followeeId);
 */

public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        long[] distances = {distance(p1, p2), distance(p1, p3), distance(p1, p4), distance(p2, p3), distance(p2, p4),distance(p3, p4)};
        Arrays.sort(distances);
        if(distances[0] == 0 || distances[0] != distances[3] || distances[3] >= distances[4] || distances[4] != distances[5]) return false;
        return true;
      
    }
    
    private long distance(int[] p1, int[] p2){
        long x = (long)(p1[0] - p2[0]);
        long y = (long)(p1[1] - p2[1]);
        return x*x + y*y;
    }
public class Solution {
    public String validIPAddress(String IP) {
        if(isValidIPv4(IP)) return "IPv4";
        else if(isValidIPv6(IP)) return "IPv6";
        return "Neither";
        
    }
    private boolean isValidIPv4(String s){
        if(s == null || s.length() == 0 || s.charAt(0) == '.' || s.charAt(s.length() -1) == '.' || s.length() > 15){
            return false;
        }
        String[] toString = s.split("\\.");
        if(toString.length != 4) return false;
        for(String ss : toString){
            if(ss.length() > 3 || ss.length() == 0) return false;
            if(ss.length() > 1 && ss.charAt(0) == '0') return false;
            for(char c: ss.toCharArray()){
                if(!Character.isDigit(c)) return false;
                
            }
            int num = Integer.parseInt(ss);
            if(num > 255) return false;
        }
        return true;
    }
    private boolean isValidIPv6(String s){
        if(s == null || s.length() == 0 || s.charAt(0) == ':' || s.charAt(s.length() -1) == ':' || s.length() > 39){
            return false;
        }
        String[] toString = s.split(":");
        if(toString.length != 8) return false;
        for(String ss : toString){
            if(ss.length() == 0 || ss.length() > 4) return false;
          for(char c: ss.toCharArray()){
              if(!Character.isDigit(c)){
                  if( c > 'F' && c < 'a') return false;
                  if( c > 'f' || c < 'A') return false;
              }
          }
        }
        return true;
        
    }
}

class Node{
    int data;
    Node left, right;
    public Node(int item){
        data = item;
        left = right = null;
    }  
}

public class BinaryTree{
    Node root;
    static boolean v1 = false, v2 = false;
    Node findLCAUtil(Node node, int n1, int n2){
        if(node == null) return null;
        if(node.data == n1) {
            v1 = true;
            return node;
        }
        if(node.data == n2) {
            v2 = true;
            return node;
        }
        Node left_lca = findLCAUtil(node.left, n1, n2);
        Node right_lca = findLCAUtil(node.right, n1, n2);
        if(left_lca != null && right_lca != null)
            return node;
        return (left_lca != null)? left_lca : right_lca;    
    }
    Node findLCA(int n1, int n2){
        v1 = false;
        v2 = false;
        Node lca = findLCAUtil(root, n1, n2);
        if(v1 && v2) return lca;
        return null;

    }
}

public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int maxLength = Math.max(v1.length, v2.length);
        for(int i = 0; i < maxLength; i++){
            Integer val1 = i < v1.length ? Integer.parseInt(v1[i]) : 0;
            Integer val2 = i < v2.length ? Integer.parseInt(v2[i]) : 0;
         
            int comp = val1.compareTo(val2); // Integer can be used here for compareTo, int can not be used instead.
            if(comp != 0){
                return comp;
            }
        }
        return 0;
        
    }

 public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        if(desiredTotal <= maxChoosableInteger) return true;
        if(maxChoosableInteger*(maxChoosableInteger + 1) / 2 < desiredTotal) return false;
        return help(maxChoosableInteger, desiredTotal, 0, new HashMap<>());
        
    }
    private boolean help(int n, int t, int state, HashMap<Integer, Boolean> map){
        if(map.containsKey(state)) return map.get(state);
        for(int i = 0; i < n; i++){
            if((1 << i & state) != 0) continue;
            if(t <= i+ 1 || !help(n, t - (i+1), state |(1 << i), map)){
                 map.put(state, true);
                    return true;
            }
               
        }
        map.put(state, false);
        return false;
    }

// Definition of dp[i][j]: minimum number of money to guarantee win for subproblem [i, j].

// Target: dp[1][n]

// Corner case: dp[i][i] = 0 (because the only element must be correct)

// Equation: we can choose k (i<=k<=j) as our guess, and pay price k. After our guess, the problem is divided into two subproblems. Notice we do not need to pay the money for both subproblems. We only need to pay the worst case (because the system will tell us which side we should go) to guarantee win. So dp[i][j] = min (i<=k<=j) { k + max(dp[i][k-1], dp[k+1][j]) }
public int getMoneyAmount(int n) {
        if(n == 1) return 0;
        int[][] dp = new int[n+2][n+2];
        for(int len = 1; len < n; len++){
            for(int i = 1; i + len <= n; i++){
                int j = i + len;
                dp[i][j] = Integer.MAX_VALUE;
                for(int k = i; k <= j; k++){
                    dp[i][j] = Math.min(dp[i][j], Math.max(dp[i][k-1], dp[k+1][j]) + k);
                }
            }
        }
        return dp[1][n];
        
    }
 
public class NumArray {
    class RTN{
        int start; 
        int end;
        int sum;
        RTN left;
        RTN right;
    public RTN(int start, int end) {
        this.start = start;
        this.end = end;
        this.sum = 0;
        this.left = null;
        this.right = null;
    }
    }
    private RTN root;
    private int[] nums;
    public NumArray(int[] nums) {
        this.nums = new int[nums.length];
        for(int i = 0; i < nums.length; i++){
            this.nums[i] = nums[i];
        }
        root = buildTree(nums, 0, nums.length -1);
        
    }
    private RTN buildTree(int[] nums, int start, int end){
        if(start > end) return null;
        else{
            RTN root = new RTN(start, end);//Here is to initialize a new node;
            if(start == end) {
                root.sum = nums[start];

            }else{
                int mid = start + (end - start)/2;
                root.left = buildTree(nums, start, mid);
                root.right = buildTree(nums, mid+ 1, end);
                root.sum = root.left.sum + root.right.sum;
            }
            return root;
        }
    }

    
    public void update(int i, int val) {
        int diff = val - this.nums[i];
        this.nums[i] = val;
        update(root, i, diff);
    }
    private void update(RTN root, int i, int diff){
        if(root == null || i > root.end || i < root.start) return;
        if( i >= root.start && i <= root.end) root.sum += diff;
        update(root.left, i, diff);
        update(root.right, i, diff);
    }
    
    public int sumRange(int i, int j) {
        if(root == null || i > j || (i <0  && j < 0)) return 0;
        return sumRange(root, i, j);
        
    }
    
    private int sumRange(RTN root, int i, int j) {
        if(root == null || i > root.end || j < root.start) return 0;
        if( i <= root.start && j >= root.end) return root.sum;
        else return sumRange(root.left, i,j) + sumRange(root.right, i, j);
    }
}

public void solve(char[][] board){
    if(board == null || board.length == 0 || board[0].length == 0) return;
    int m = board.length;
    int n = board[0].length;
    for(int j = 0; j < n ; j++){
        if(board[0][j] == 'O'){
            bfs(board, 0, j);
        }
    }
    for(int j = 0; j < n; j++){
        if(board[m-1][j] == 'O'){
            bfs(board, m-1, j);
        }
    }
    for(int i = 0; i < m; i++){
        if(board[i][0] == 'O'){
            bfs(board, i, 0);
        }
    }
    for(int i = 0; i < m; i++){
        if(board[i][n-1] == 'O'){
            bfs(board, i, n-1);
        }
    } 

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(board[i][j] == '1'){
                board[i][j] = 'O';
            }
            if(board[i][j] == 'O'){
                board[i][j] = 'X';
            }
        }
    }   
private void bfs(char[][] board, int i, int j){
    int m = board.length;
    int n = board[0].length;
    board[i][j] = '1';
    Queue<int[]> queue = new LinkedList<int[]>();
    q.add(new int[]{i, j});
    while(!q.isEmpty()){
        if(i > 0 && board[i-1][j] == 'O'){
            board[i-1][j] == '1';
            queue.add(new int[]{i-1, j});
        }
        if(i+1 < m && board[i+1][j] == 'O'){
            board[i+1][j] == '1';
            queue.add(new int[]{i+1, j});
        }
        if(j - 1 >= 0 && board[i][j-1] == 'O'){
            board[i][j-1] == '1';
            queue.add(new int[]{i, j-1});
        }
        if(j + 1 < n && board[i][j+1] == 'O'){
            board[i][j+1] == '1';
            queue.add(new int[]{i, j+1});
        }
    }

}

 public String fractionToDecimal(int numerator, int denominator) {
    if(numerator == 0) return "0";
    long num = Math.abs((long) numerator);
    long denom = Math.abs((long) denominator);
    StringBuilder sb = new StringBuilder();
    String integral = String.valueOf(num/denom);
    if((numerator != 0) && ((numerator < 0) ^ (denominator < 0))){
        integral = "-" + integral;
    }
    sb.append(integral);
    num %= denom;
    if(num == 0) return sb.toString();
    sb.append(".");
    Map<Long, Integer> map = new HashMap<>();
    while(num != 0){
        map.put(num, sb.length());
        num *= 10;
        sb.append(num/denom);
        num %= denom;
        if(map.containsKey(num)){
            sb.insert(map.get(num), "(");
            sb.append(")");
            break;
        }
    }
    return sb.toString();
}





