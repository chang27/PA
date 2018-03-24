// regex:
public boolean isMatch(String s, String p) {
that is back tracking
}
//dp solution:



public class Solution {
    public int jump(int[] nums) {
        int curmax = 0;
        int lastmax = 0;
        int step = 0;
        int n = nums.length - 1;
        for(int i = 0; i<= n; i++){

            if(i > lastmax){
                lastmax = curmax;
                step++;
                if(lastmax >= n) return step;
            }
            curmax = Math.max(curmax, i + nums[i]);
        }
        if(curmax < n-1) return 0;
        return step;
    }
}
//-------------------------------------------------------
public int jump(int[] A) {
    int count = 0, max = 0;
    for (int i = 0, nextMax = 0; i <= max && i < A.length - 1; i++) {
        nextMax = Math.max(nextMax, i + A[i]);
        if (i == max) {
            max = nextMax;
            count++;
        }
    }
    // if there is no way to get to the end, return -1
    return max >= A.length - 1 ? count : -1;
}

//word  break2

public class Solution {
    HashMap<String,List<String>> map = new HashMap<String,List<String>>();
    public List<String> wordBreak(String s, List<String> wordDict) {
        List<String> res = new ArrayList<String>();
       if(s.length() == 0 || s == null) return res;
       if(map.containsKey(s)) return map.get(s);
       if(wordDict.contains(s)) res.add(s);  //key step
       for(int i = 1; i < s.length(); i++){
           String t = s.substring(i);
           if(wordDict.contains(t)){
               List<String> tmp = wordBreak(s.substring(0, i), wordDict);
               if(tmp.size() != 0){
                   for(int j = 0; j < tmp.size(); j++){
                       res.add(tmp.get(j) + " " + t);
                   }
               }
           }
       }
       map.put(s, res);
       return res;
    }
}

public class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<String, List<String>> map = new HashMap<>();
        return helper();
    }
    public List<String> helper(String s, List<String> Dict, Map<String, List<String>> map){
        List<String> res = new ArrayList<String>();
        if(s.length() == 0 || s== null){
            return res;
        }
        if(map.containsKey(s)){
            return map.get(s);
        }
        for(int i = 1; i < s.length() + 1; i++){
            String sub = s.substring(0, i);
            if(Dict.contains(sub)){
                if(i == s.length()) res.add(sub); //key step
                else{
                    List<String> tmp = helper(s.substring(i), Dict, map);
                    for(String item: tmp){
                        res.add(sub + " " + item);
                    }
                }
            }
        }
        map.put(s, res);
        return res;

    }
}

public class Solution {

        HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();

public List<String> wordBreak(String s, List<String> wordDict) {
    if (map.containsKey(s))
        return map.get(s);

    ArrayList<String> res = new ArrayList<String>();
    if (s.length() == 0) {
        //res.add("");
        return res;
    }
    for (String word : wordDict) {
        if (s.startsWith(word)) {
            if (s.length() == word.length()) res.add(word);
            else{
            List<String> sublist = wordBreak(s.substring(word.length()), wordDict);
            for (String sub : sublist)
                res.add(word + (sub.isEmpty() ? "" : " ") + sub);
        }
        }
    }
    map.put(s, res);
    return res;
}
}

//word ladder

public class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if(!wordList.contains(endWord)) return 0;
        Set<String> beginSet = new HashSet<>();
        Set<String> endSet = new HashSet<>();
        Set<String> visited = new HashSet<>();
        beginSet.add(beginWord);
        endSet.add(endWord);
        int len = 1;
        while(!beginSet.isEmpty() && !endSet.isEmpty()){
            if(beginSet.size() > endSet.size()){
                Set<String> set = beginSet;
                beginSet = endSet;
                endSet = set;
            }
            Set<String> temp = new HashSet<>();
            for(String s: beginSet){
                for(String t: wordList){
                    if(isValid(s,t)){
                        if(endSet.contains(t)) return len + 1;
                        else if(!visited.contains(t)){
                            temp.add(t);
                            visited.add(t);
                        }
                        
                    }
                }
            }
            beginSet = temp;
            len++;
        }
        return 0;
    }
    private boolean valid(String a, String b){
        int cnt = 0;
        for(int i = 0; i < a.length(); i++){
            if(a.charAt(i) != b.charAt(i)) cnt++;
            if (cnt > 1) return false;
        }
        return true;
    }
}
public class Solution {
    public List<List<String>> findLadders(String start, String end, Set<String> dict){

    }
}

class WordNode{
    String word;
    int numSteps;
    WordNode pre;
    public WordNode(String word, int numSteps, WordNode pre){
        this.word = word;
        this.numSteps = numSteps;
        this.pre = pre;
    }
}
public class Solution{
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList){
        List<List<String>> result = new ArrayList<List<String>>();
        if(!wordList.contains(endWord)) {return result;}
        LinkedList<WordNode> queue = new LinkedList<WordNode>();
        queue.add(new WordNode(beginWord, 1, null));
        int minStep = 0;
        HashSet<String> visited = new HashSet<String>();
        HashSet<String> unvisited = new HashSet<String>();
        unvisited.addAll(wordList);
        int preNumSteps = 0;
        while(!queue.isEmpty()){
            WordNode top = queue.remove();
            String word = top.word;
            int curNumSteps = top.numSteps;
            if(word.equals(endWord)){
                if(minStep == 0){
                    minStep = top.numSteps;
                }

            if(top.numSteps == minStep && minStep != 0){
                ArrayList<String> t = new ArrayList<String>();
                t.add(top.word);
                while(top.pre != null){
                    t.add(0, top.pre.word);
                    top = top.pre;
                }
                result.add(t);
                continue;
            }
        }
        if(preNumSteps < curNumSteps){
            unvisited.removeAll(visited);
        }
        preNumSteps = curNumSteps;

        char[] arr = word.toCharArray();
        for(int i = 0; i < arr.length;i++){
            for( char c = 'a'; c <= 'z'; c++){
                char temp = arr[i];
                if(arr[i] != c){
                    arr[i] = c;
                }
                String newWord = new String(arr);
                if(unvisited.contains(newWord)){
                    queue.add(new WordNode(newWord, top.numSteps +1, top));
                    visited.add(newWord);
                }
                arr[i] = temp;
            }
        }

    }
    return result;
}
}
// here is TLE, because the list<String>wordlist, need to change it to hashset to save time.

public class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
       
    List<List<String>> res = new ArrayList<List<String>>();
        if(!wordList.contains(endWord)) return res;
        Map<String, Integer> distance = new HashMap<String, Integer>();
        Map<String, List<String>> neighbors = new HashMap<String, List<String>>();
        wordList.add(beginWord);

        bfs(neighbors, distance,beginWord,endWord, wordList);
        List<String> solution = new ArrayList<>();
        dfs(beginWord, endWord, wordList, res, distance, neighbors, solution);
        return res;

        }
    
    void bfs(Map<String, List<String>> neighbors, Map<String, Integer> distance, String start, String end, List<String> dict){
        for(String str : dict){
            neighbors.put(str, new ArrayList<String>());
        }
        
        Queue<String> queue = new LinkedList<String>();
        queue.offer(start);
        distance.put(start, 1);
        
        while(!queue.isEmpty()){
            int count = queue.size();
            boolean found = false;
            for(int i = 0; i < count; i++){
                String cur = queue.poll();
                int curDistance = distance.get(cur);
                List<String> nei = getNeighbors(cur, dict);
               // List<String> final = new ArrayList<String>();
                for(String n : nei){
                    if(!distance.containsKey(n)){
                        distance.put(n, curDistance+1);
                        neighbors.get(cur).add(n);
                        if(end.equals(n)){
                            found = true;
                        }
                        else{
                            queue.offer(n);
                        }
                    }else if(distance.get(n) == curDistance +1){
                        neighbors.get(cur).add(n);
                    }
                }

            }
            if(found){
                break;
            }

        }
    }
        private List<String> getNeighbors(String node, List<String> dict){
            List<String> res = new ArrayList<>();
            char[] arr = node.toCharArray();
            
                for(int i = 0; i < arr.length; i++){
                    for(char ch = 'a'; ch < 'z'; ch++){
                    if(arr[i] == ch) continue;
                    char temp = arr[i];
                    arr[i] = ch;
                    if(dict.contains(String.valueOf(arr))){
                        res.add(String.valueOf(arr));
                    }
                    arr[i] = temp;
                }
            }
            return res;
        }

    private void dfs(String start, String end, List<String> dict, List<List<String>> res, Map<String, Integer> distance, Map<String, List<String>> neighbors, List<String> solution){
        solution.add(start);
        if(start.equals(end)){
           // Collections.reverse(solution);
            res.add(new ArrayList<String>(solution));
        }else{
            for(String next: neighbors.get(start)){
                if(distance.get(next) == distance.get(start)+1){
                    dfs(next, end, dict, res, distance,neighbors,solution);
                }
            }
        }
        solution.remove(solution.size() -1);

    }   
}

public static int calculate(String s) {
    int len = s.length();
    int result = 0;
    int sign = 1;

    Stack<Integer> stack = new Stack<Integer>();
    for(int i = 0; i < len; i++){
        if(Character.isDigit(s.charAt(i))) {
            int num = s.charAt(i) - '0';
            while(i+1 < len && Character.isDigit(s.charAt(i+1))){
                num = num*10 + s.charAt(i+1) - '0';
                i++;
            }
            result += num*sign;
        }
        else if(s.charAt(i) == '+'){
            sign = 1;
        }
        else if(s.charAt(i) == '-'){
            sign = -1;
        }
        else if(s.charAt(i) == '('){
            stack.push(result);
            stack.push(sign);
            result = 0;
            sign = 1;
        }else if(s.charAt(i) == ')'){
            result = result * stack.pop() + stack.pop();
        }
    }
    return result;
}

public int scheduleCourse(int[][] courses){
    int n = courses.length;
    if(n == 0) return 0;
    Arrays.sort(courses, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            return a[1] == b[1] ? a[0] - b[0];
        }
    });
    PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
    int start = 0;
    for(int[] course : courses) {
        start += course[0];
        pq.offer(course[0]);
        if(start > course[1]){
            start -= pq.poll();
        }
    }
    return pq.size();

}

public static int longestValidParentheses(String s){
    Stack<int[]> stack = new Stack<int[]>();
    int res = 0;
    for(int i = 0; i < s.length(); i++){
        if(s.charAt(i) == '('){
            int[] a = {i, 0};
            stack.push(a);
        }
        else{
            if(stack.isEmpty() || stack.peek()[1] == 1){
                int[] b = {i, 1};
                stack.push(b);
            }else{
                stack.pop();
                int cur = 0;
                if(stack.isEmpty()) {cur = i + 1;}
                else {cur = i - stack.peek()[0];}
                res = Math.max(res, cur);
            }
        }
    }
    return res;
}


public List<String> removeInvalidParentheses(String s){
    List<String> ans = new ArrayList<>();
    help(s, ans, 0, 0, new char[]{'(', ')'});
    return ans;
}

private void help(String s, List<String> res, int si, int bi, char[] par){
    int cnt = 0;
    for(int i = si; i < s.length(); i++){
        if(s.charAt(i) == par[0]) cnt++;
        else if(s.charAt(i) == par[1]) cnt--;

        if(cnt >= 0) continue;
        for(int j = bi; j <= i ; j++){
            if(s.charAt(j) == par[1] && (j == 0 || s.charAt(j -1) != par[1])){
                help(s.substring(0, j) + s.substring(j+1), res, i, j, par);
            }
        }
        return;
    }
    String reverse = new StringBuilder(s).reverse().toString();
    if(par[0] == '('){
        help(reverse, res, 0, 0, new char[]{')', '('});
    }else{
        res.add(reverse);
    }
}

public List<String> removeInvalidParentheses(String s) {
    List<String> res = new ArrayList<>();
    if(s == null) return res;
    Set<String> visited = new HashSet<>();
    Queue<String> queue = new LinkedList<>();
    queue.add(s);
    visited.add(s);
    boolean found = false;
    while(! queue.isEmpty()){
        s = queue.poll();
        if(isValid(s)){
            res.add(s);
            found = true;
        }
        if(found) continue;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) != '(' && s.charAt(i) != ')'){
                continue;
            }
            String t = s.substring(0, i) + s.substring(i+1);
            if(!visited.contains(t)){
                queue.add(t);
                visited.add(t);
            }
        }

    }
    return res;

}

boolean isValid(String s) {
    int cnt = 0;
    for(int i = 0; i < s.length(); i++){
        char c = s.charAt(i);
        if(c == '(') cnt++;
        if(c == ')') cnt--;
        if(c < 0) return false;
    }
    return cnt == 0;
}
// this is try to find all the valid parentheses in a string. there should be another loop in the main function.
   public List<String> removeInvalidParentheses(String s) {
          List<String> ans = new ArrayList<>();
    help(s, ans, 0, new char[]{'(', ')'});
        if(ans.isEmpty()){
            ans.add("");
        }
        HashSet<String> set = new HashSet<>();
        for(String ss : ans){
            set.add(ss);
        }
        
    return new ArrayList<>(set);
        
    }
    
    private void help(String s, List<String> res,int start, char[] par){
        if(s == null || s.length() == 0) return;
        int cnt = 0;
        for(int i = start; i < s.length(); i++){
            if(isValid(s.substring(0, i+1))) res.add(s.substring(0, i+1));
        for(int j = 0; j <= i ; j++){
            if(s.charAt(j) == par[1] && (j == 0 || s.charAt(j -1) != par[1])){
               // help(s.substring(0, j) + s.substring(j+1, i+1), res, 0, par);
                if (isValid(s.substring(0, j) + s.substring(j+1, i+1))) res.add(s.substring(0, j) + s.substring(j+1, i+1));
            }
            
        }
        
        }
       // return;
    }
    private boolean isValid(String s){
        int cnt = 0;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == '(') cnt++;
            if(s.charAt(i) == ')'){
                if(cnt ==0) return false;
                cnt --;
            }
        }
        return cnt == 0;
    }
 public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        if(s == null) return res;
        Set<String> visited = new HashSet<>();
        Queue<String> q = new LinkedList<String>();
        q.add(s);
        visited.add(s);
        while(!q.isEmpty()){
            String cur = q.poll();
            if(cur.length() == 0) break;
            if(isValid(cur)) res.add(cur);
            for(int i = 0; i < cur.length(); i++){
                if(cur.charAt(i) != '(' && cur.charAt(i) != ')'){
                    continue;
                }
                String sub = cur.substring(0, i) + cur.substring(i+1);
                if(!visited.contains(sub)){
                    q.add(sub);
                    visited.add(sub);
                }
                
            }
        }
        return res;
    }

    private boolean isValid(String s){
        int cnt = 0;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == '(') cnt++;
            if(s.charAt(i) == ')'){
                if(cnt ==0) return false;
                cnt --;
            }
        }
        return cnt == 0;
    }
public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode pre = null;
        while(!stack.isEmpty() || p != null ){
            if(p!= null){
                stack.push(p);
                p = p.left;
            }else{
                p = stack.peek();
                if(p.right != null && pre != p.right){
                    p = p.right;
                }else{
                    
                    res.add(p.val);
                    stack.pop();
                    pre = p;
                    p = null;
                }
            }
        }
        return res;
        
    }


    public List<Integer> postorderTraversal(TreeNode root){
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        List<TreeNode> peeked = new ArrayList<>();
        while(!stack.isEmpty()){
            TreeNode cur = stack.peek();
            if((cur.left == null && cur.right == null) || peeked.contains(cur)){
                res.add(stack.pop().val);
            }else{
                if(cur.right != null){
                    stack.push(cur.right);
                }
                if(cur.left != null){
                    stack.push(cur.left);
                }
                peeked.add(cur);
            }
        }
        return res;
    }


    public class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        if(n <= 0) return res;
        int[] col = new int[n];
        dfs(n, res, 0, col);
        return res;
        
    }
    
    private void dfs(int n, List<List<String>> res, int row, int[] col){
        if(row == n){
            List<String> unit = new ArrayList<>();
            for(int i = 0; i < n; i++){
                StringBuilder sb = new StringBuilder();
                for(int j = 0; j < n; j++){
                    if(col[i] == j){
                        sb.append("Q");
                    }
                    else sb.append(".");
                }
                unit.add(sb.toString());
            }
            res.add(unit);
            
        }else{
            for(int i = 0; i < n; i++){
                col[row] = i;
                if(isValid(row, col)){
                    dfs(n, res, row+1, col);
                }
            }
        }
    }
    private boolean isValid(int row, int[] col){
        for(int i = 0; i < row; i++){
            if(col[i] == col[row] || Math.abs(col[i] - col[row]) == row - i){
                return false;
            }
        }
        return true;
    }
}

public class Solution {
    int res = 0; // IMPORTANT HERE must be a global variable!!!
    public int totalNQueens(int n) {
        if(n <= 0) return 0;
        
        int[] col = new int[n];
        dfs(n, 0, col);
        return res;
        
    }
    private void dfs(int n,int row, int[] col){
        if(row == n){
            res += 1;
        }else{
            for(int i = 0; i < n; i++){
                col[row] = i;
                if(isValid(row, col)){
                    dfs(n,row+1, col);
                }
            }
        }
        
    }
    
    private boolean isValid(int row, int[] col){
        for(int i = 0; i< row; i++){
            if(col[i] == col[row] || Math.abs(col[i] - col[row]) == row - i){
                return false;
            }
        }
        return true;
    }
}

public int[] smallestRange(List<List<Integer>> nums) {
    PriorityQueue<int[]> pq = new PriorityQueue<>((a,b) -> a[1] - b[1]);
    int max = Integer.MIN_VALUE;
    for(int i = 0; i < nums.size(); i++){
        pq.offer(new int[]{i, nums.get(i).get(0), 0});
        max = Math.max(max, nums.get(i).get(0));
    }
    int minRange = Integer.MAX_VALUE;
    int start = -1;
    while(pq.size() == nums.size()){
        int[] small = pq.poll();
        if(max - small[1] < minRange){
            minRange = max - small[1];
            start = small[1];
            if(minRange == 0) break;
        }
        if(small[2] + 1 < nums.get(small[0]).size()){
            small[2] += 1;
            small[1] = nums.get(small[0]).get(small[2]);
            pq.offer(small);
            if (small[1] > max) max = small[1];
        }

    }
    return new int[]{start, start + minRange};
}

public String minWindow(String s, String t) {
        if(s.length() < t.length()) return "";
        if(s.equals(t)) return s;
        int[] map = new int[128];
        for(char c: t.toCharArray()){
            map[c]++;
        }
        int start = 0, end = 0, minStart = 0, minLen = Integer.MAX_VALUE, cnt = t.length();
        while(end < s.length()){
            char c1 = s.charAt(end);
            if(map[c1] > 0) cnt--;
            map[c1]--;
            end++;
            while(cnt == 0){
                if(minLen > end - start){
                    minLen = end - start;
                    minStart = start;
                }
                char c2 = s.charAt(start);
                map[c2]++;
                if(map[c2] > 0) cnt++;
                start++;
            }
    }
        return minLen == Integer.MAX_VALUE? "" : s.substring(minStart, minStart + minLen);
}
public int lengthOfLongestSubstringKDistinct(String s, int k){
    int[] map = new int[128];
    int start = 0, end = 0, maxLen = Integer.MAX_VALUE, cnt = 0;
    while(end < s.length()){
        char c1 = s.charAt(end);
        if(map[c1] == 0) cnt++;
        map[c1]++;
        end++;
        while(counter > k){
            char c2 = s.charAt(start);
            if(map[c2] == 1) cnt--;
            map[c2]--;
            start ++;
        }
        maxLen = Math.max(maxLen, end - start);
    }
    return maxLen;

}

public int minSubArrayLen(int s, int[] nums) {
        if(nums == null || nums.length ==0) return 0;
        int[] map = new int[10];
        int start = 0, end = 0, minLen = Integer.MAX_VALUE, sum = 0;
        while(end < nums.length){
            sum += nums[end];
            end++;
            while(sum >= s){
                minLen = Math.min(minLen, end - start);
                sum -= nums[start];
                start++;    
            }
        }
        return minLen == Integer.MAX_VALUE? 0: minLen;
    }

 

 public class Solution {
    public int maxCoins(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int n = nums.length;
        int[] iNums = new int[n+2];
        int[][] mem = new int[n+2][n+2];
        for(int i= 1; i <= n; i++){
            iNums[i] = nums[i-1];
        }
        iNums[0] = 1;
        iNums[n+1] = 1;
        
     
     return help(iNums, mem, 1, n);
}
private int help(int[] nums, int[][] mem, int start, int end){
    if(start > end) return 0;
    if(mem[start][end] > 0) return mem[start][end];
    int max = 0;
    for(int i = start; i <= end; i++){
        int res = nums[start - 1]*nums[i]*nums[end+1] + help(nums, mem, start, i- 1) + help(nums, mem, i+1, end);
        if(res > max){
            max = res;
        }
    }
    mem[start][end] = max;
    return max;
} 
}

public class Solution {
    public int maxCoins(int[] nums) {
        if(nums == null || nums.length == 0) return 0;
        int n = nums.length;
        int[] iNums = new int[n+2];
        for(int i = 0; i <= n-1; i++){
            iNums[i+1] = nums[i];
        }
        iNums[0] = 1;
        iNums[n+1] = 1;
        int[][] dp = new int[n+2][n+2];
        for(int len= 1; len <= n; len++){
            for(int i = 1; i+len < n+2; i++){
                for(int j = i; j < i+len; j++){// totally calculate n values;
                   int val = iNums[i-1] * iNums[j]* iNums[i+len];
                   dp[i][i+len-1] = Math.max(dp[i][i+len-1], val + dp[i][j-1] +dp[j+1][i+len-1]); 
                }
            }
        
        }  
     return dp[1][n];
}
}

public boolean comparison(String str, String pattern){
    int s = 0, p = 0, match = 0, starIndex = -1;
    while(s < str.length()){
        if(p < pattern.length() &&(pattern.charAt(p) == '?'||str.charAt(s) == pattern.charAt(p))){
            s++;
            p++;
        }else if(p < pattern.length() && pattern.charAt(p) == '*'){
            starIndex = p;
            match = s;
            p++;
        }
        else if(starIndex != -1){
            p = starIndex + 1;
            s = ++match;
        }else return false;
    }
    while( p < pattern.length && pattern.charAt(p) == '*'){
        p++;
    }
    return p == pattern.length();

}



    public int maxProfit(int k, int[] prices) {
        if(prices == null || prices.length <= 1) return 0;
        int n = prices.length;
        if(k >= n/2) {
            int profit = 0;
            for(int i = 1; i < prices.length;i++){
                if(prices[i] - prices[i-1] > 0) profit += prices[i] - prices[i-1];
            }
            return profit;
        }
        int[][]dp = new int[k+1][n];
        
        for(int i = 1; i <= k; i++) {
            int max = -prices[0];
            for(int j = 1; j<n; j++){
                dp[i][j] = Math.max(dp[i][j-1], prices[j] + max);
                max = Math.max(max, dp[i-1][j] - prices[j]);
            }
        }
        return dp[k][n-1];
    }

     public int maxProfit(int[] prices) {
        if(prices == null || prices.length <= 1) return 0;
        int[] left = new int[prices.length];
        int[] right = new int[prices.length];
        int min = prices[0];
        for(int i = 1;  i < prices.length; i++){
            left[i] = Math.max(left[i-1], prices[i] - min);
            min = Math.min(prices[i], min);
        }
        int max = prices[prices.length - 1];
        for(int i = prices.length -2; i >= 0; i--){
            right[i] = Math.max(right[i+1], max - prices[i]);
            max = Math.max(max, prices[i]);
        }
        int maxx = 0;
        for(int i = 0; i < prices.length; i++){
            maxx = Math.max(maxx, left[i] + right[i]);
        }
        return maxx;
        
    }

        public int maxProfit(int[] prices) {
        if(prices == null || prices.length <= 1) return 0;
        int buy = -prices[0];
        int sell = 0;
        int presell = sell;
        int prebuy = 0;
        for(int i = 1; i < prices.length; i++){
            prebuy = buy;
            buy = Math.max(buy, presell - prices[i]);
            presell = sell;
            sell = Math.max(presell, prebuy + prices[i]);
        }
        return sell;
    }

    class SummaryRanges {
    private TreeMap<Integer, Interval> treeMap;
    /** Initialize your data structure here. */
    public SummaryRanges() {
        treeMap = new TreeMap<>();
    }
    
    public void addNum(int val) {
        if(treeMap.containsKey(val)) return;
        Integer low = treeMap.lowerKey(val);
        Integer high = treeMap.higherKey(val);
        if( low != null && high != null && treeMap.get(low).end + 1 == val && high == val + 1){
            treeMap.put(low, new Interval(low, treeMap.get(high).end));
            treeMap.remove(high);
        }
        else if(low != null && treeMap.get(low).end + 1 >= val){
            treeMap.get(low).end = Math.max(treeMap.get(low).end, val);
        }else if(high != null && high == val+1){
            treeMap.put(val, new Interval(val, treeMap.get(high).end));
            treeMap.remove(high);
        }else{
            treeMap.put(val, new Interval(val, val));
        }
        
    }
    
    public List<Interval> getIntervals() {
        return new ArrayList<>(treeMap.values());
    }
}


    public int findRotateSteps(String ring, String key) {
        
        Map<Character, List<Integer>> map = new HashMap<>();
        for(int i = 0; i < ring.length(); i++){
            if(!map.containsKey(ring.charAt(i))){
                map.put(ring.charAt(i), new ArrayList<>());
            }
            map.get(ring.charAt(i)).add(i);
        }
        int[][] dp = new int[key.length()][];
        dp[0] = new int[map.get(key.charAt(0)).size()];
        for(int i = 0; i < dp[0].length ; i++){
            int j = map.get(key.charAt(0)).get(i);
            dp[0][i] = Math.min(j- 0, ring.length() - j);
        }
        for(int i = 1; i < key.length(); i++){
            dp[i] = new int[map.get(key.charAt(i)).size()];
            for(int j = 0; j < dp[i].length; j++){
                dp[i][j] = Integer.MAX_VALUE;
                int b = map.get(key.charAt(i)).get(j);
                for(int k = 0; k < dp[i-1].length; k++){
                    int a = map.get(key.charAt(i -1)).get(k);
                    dp[i][j] = Math.min(dp[i][j],dp[i-1][k] + Math.min(Math.abs(b-a), ring.length() - Math.abs(b-a)));
                }
                
            }
        }
        int min = dp[key.length() -1][0];
        for(int i = 1; i < dp[key.length() -1].length; i++){
            min = Math.min(min, dp[key.length() -1][i]);
        }
        return min+ key.length();
    }


public int findRotateSteps(String ring, String key){
    int n = ring.length();
    int m = key.length();
    int[][] dp = new int[m+1][n];
    for(int i = m -1; i >= 0; i--){
        for(int j = 0; j < n; j++) {
            dp[i][j] = Integer.MAX_VALUE;
            for(int k = 0; k < n; k++) {
                if(ring.charAt(k) == key.charAt(i)){
                    int diff = Math.abs(k - j);
                    int step = Math.min(diff, n - diff);
                    dp[i][j] = Math.min(dp[i][j], step + dp[i+1][k]);
                }
            }
        }
    }
}
// MAXCOUNT is needed instead of Integer.MAX_VALUE;
// "#" at the end of the string is needed in case of "WWWW" and j will be equal to string.length;

class Solution {
    int MAXCOUNT = 6;
    public int findMinStep(String board, String hand) {
    int[] handCount = new int[26];
    for (int i = 0; i < hand.length(); ++i) ++handCount[hand.charAt(i) - 'A'];
    int rs = helper(board + "#", handCount);  // append a "#" to avoid special process while j==board.length, make the code shorter.
    return rs == MAXCOUNT ? -1 : rs;
}
private int helper(String s, int[] h) {
  // s = removeConsecutive(s);     
    if (s.equals("#")) return 0;
    int  rs = MAXCOUNT, need = 0;
    for (int i = 0, j = 0 ; j < s.length(); ++j) {
        if (s.charAt(j) == s.charAt(i)) continue;
        need = 3 - (j - i);     //balls need to remove current consecutive balls.
        if(need <= 0) need = 0;
        if (h[s.charAt(i) - 'A'] >= need) {
            h[s.charAt(i) - 'A'] -= need;
            rs = Math.min(rs, need + helper(s.substring(0, i) + s.substring(j), h));
            h[s.charAt(i) - 'A'] += need;
        }
        i = j;
    }
    return rs;
}
    
    private String removeConsecutive(String board) {
    for (int i = 0, j = 0; j < board.length(); ++j) {
        if (board.charAt(j) == board.charAt(i)) continue;
        if (j - i >= 3) return removeConsecutive(board.substring(0, i) + board.substring(j));
        else i = j;
    }
    return board;
}
        
    
}

private TreeNode first = null;
private TreeNode second = null;
private TreeNode pre = null;

public void recoverTree(TreeNode root){
    inorder(root);
    int tmp = first.val;
    first.val = second.val;
    second.val =  tmp;
}
private void inorder(TreeNode root){
    if(root == null) return;
    inroder(root.left);
    if(pre != null && pre.val > root.val){
        if(first == null){
            first = pre;
            second = root;
        }else{
            second = root;
        }
    }
    pre = root;
    inorder(root.right);
}


private void recoverTree(TreeNode root){
    if(root == null) return;
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = root;
    TreeNode pre = null, first = null, second = null;
    while(cur != null || !stack.isEmpty()){
        if(cur != null){
            stack.push(cur);
            cur = cur.left;
        }else{
            cur = stack.pop();
            if(pre != null && pre.val >= cur.val){
                if(first == null){
                    first = prev;
                }
                if(first != null){
                    second = cur;
                }
            }
            pre = cur;
            cur = cur.right;
        }
        
        }
        int tmp = first.val;
        first.val = second.val;
        secong.val = tmp;
}
public void recoverTree(TreeNode root){
    TreeNode pre = null;
    TreeNode first = null, second = null;
    TreeNode cur = null;
    while(root != null){
        if(root.left != null){
            cur = root.left;
            while(cur.right != null && cur.right != root){
                cur = cur.right;
            }
            if(cur.right != null){
                if(pre != null && pre.val > root.val){
                    if(first == null){
                        first = pre;
                    }
                    second = root;
                }
                pre = root;
                cur.right = null;
                root = root.right;
            }else{
                cur.right = root;
                root = root.left;
            }
        }else{
            if(pre != null && pre.val > root.val){
                if(first == null){
                    first = pre;
                }
                second = root;
            }
            pre = root;
            root = root.right;
        }
    }
    if(first != null && secong !=null){
        int tmp = first.val;
        first.val = second.val;
        secong.val = tmp;
    }
}


public class Codec {
    private final String splitter = ",";
    private final String no = "X";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        helper(root, sb);
        return sb.toString();
        
    }
    private void helper(TreeNode root, StringBuilder sb){
        if(root == null) {
            sb.append(no);
            return;
        }else{
            sb.append(root.val);
            sb.append(splitter);
            helper(root.left, sb);
            sb.append(splitter);
            helper(root.right, sb);
        }
        
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        List<String> list = new LinkedList<>(Arrays.asList(data.split(splitter)));
        return buildTree(list);
    }
    
    private TreeNode buildTree(List<String> list){
        if(list == null || list.isEmpty()) return null;
        String s = list.remove(0);
        if(s.equals(no)) return null;
        TreeNode root = new TreeNode(Integer.valueOf(s));
        root.left = buildTree(list);
        root.right = buildTree(list);
        return root;
        
    }
}
public int trapRainWater(int[][] heightMap) {
    if(heightMap == null || heightMap.length <= 2 || heightMap[0].length<= 2){
        return 0;
    }
    int m = heightMap.length;
    int n = heightMap[0].length;
    PriorityQueue<int[]> q = new PriorityQueue<int[]>((a,b)-> a[2] - b[2]);
    boolean[][] visited = new boolean[m][n];
    for(int i = 0 ; i < m; i++){
        visited[i][0] = true;
        visited[i][n-1] = true;
        q.offer(new int[]{i, 0, height[i][0]});
        q.offer(new int[]{i, n-1, height[i][n-1]});

    }
    for(int i = 0; i < n; i++){
        visited[0][i] = true;
        visited[m-1][i] = true;
        q.offer(new int[]{0, i, height[0][i]});
        q.offer(new int[]{m-1, i, height[m-1][i]});
    }
    int[][] dir = new int[][]{{-1, 0}, {1, 0}, {0,1}, {0, -1}};
    int res = 0;
    while(!q.isEmpty()){
        int[] cur = q.poll();
        for(int[] d : dir){
        int row = cur[0] + d[0];
        int col = cur[1] + d[1];
        if(row > 0 && row < m-1 && col > 0 && col < n-1 && !visited[row][col]){
            visited[row][col] = true;
            if(height[row][col] < cur[2]){
                res += cur[2] - height[row][col];
            }
            q.offer(new int[]{row, col, Math.max(height[row][col], cur[2])});
        }
    }
    }
    return res;
}


public int[] maxSlidingWindow(int[] nums, int k){
    if(nums == null || nums.length == 0) return new int[0];
    int[] res = new int[nums.length - k + 1];
    int p = 0;
    LinkedList<Integer> list = new LinkedList<>();
    for(int i = 0; i < nums.length; i++){
        if(!list.isEmpty() && list.peek() < i- k + 1){
            list.poll();
        }
        while(!list.isEmpty() && nums[i] > nums[list.peekLast()]) {
            list.pollLast();
        }
        list.offer(i);
        if(i - k + 1 >= 0){
            res[p++] = nums[list.peek()];
        }
    }
    return res;
}

    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for(int num : nums){
            set.add(num);
        }
        for(int num : nums){
            // if(set.remove(num)){
            //     int pre = num - 1;
            //     int next = num +1 ;
            //     while(set.remove(pre)) pre--;
            //     while(set.remove(next)) next++;
            //     res = Math.max(res, next - pre - 1);
            // }
            int pre = num - 1;
            int next = num + 1;
            while(set.contains(pre)){
                set.remove(pre);
                pre--;
            }
            while(set.contains(next)){
                set.remove(next);
                next++;
            }
            res = Math.max(res, next - pre -1);
        
        }
        return res;
        
    }


public class Solution {
    private int max = 0;
    public int longestConsecutive(TreeNode root){
        if (root == null) return 0;
        helper(root, root.val, 0);
        return max;
    }
    public void helper(TreeNode root, int val,  int len){
        if(root.val = val){
            len++;
        }else{
            len = 1;
        }
        max = Math.max(max, len);
        helper(root.left, val +1, len);
        helper(root.right, val + 1, len);
    }
    public void dfs(TreeNode root, int val, int len){
        max = Math.max(max, len);
        if(root.left != null){
            if(root.left.val - val == 1){
                dfs(root.left, val + 1, len+1);
            }
            else dfs(root.left, val +1, 1);
        }
        if(root.right.val - val == 1) {
            dfs(root.right, val+1, len +1);
        }else dfs(root.right, val +1, 1);
    }
}
 public int findMinMoves(int[] machines) {
        if(machines == null || machines.length == 0) return 0;
        int cnt = 0;
        for(int m: machines){
            cnt += m;
        }
        if(cnt % machines.length != 0) return -1;
        int avg = cnt / machines.length;
        int result = 0;
        int c = 0;
        for(int m : machines){
            c += m - avg;
            result = Math.max(result, Math.max(Math.abs(c), m - avg));[0, 4, 0] for m-avg
           
        } loop to find the largest gap on each boundary through the array.
        // for(int m : machines){
        //     if(m > avg) left += m - avg;
        //     else if(m < avg) right += avg - m;
        //     result = Math.max(result, Math.max(Math.abs(right - left), m -avg));
           
           
        // }
        return result;
        
    }

Long.parseLong(String);
Integer.parseInt(String);
String.valueOf(...);

import java.math.*;
class Solution {
    public String smallestGoodBase(String n) {  
        long nn =Long.parseLong(n);
    long res = 0;
    for(int k = 60; k >= 2; k--){
        long s = 2; 
        long e = nn;
        while(s <= e) {
            long m = s + (e-s)/2;
            BigInteger left = BigInteger.valueOf(m);
            left = left.pow(k).subtract(BigInteger.ONE);
            BigInteger right = BigInteger.valueOf(nn).multiply(BigInteger.valueOf(m).subtract(BigInteger.ONE));
            int cmp = left.compareTo(right);
            if(cmp == 0){
                res = m;
                break;
            }else if(cmp < 0){
                s = m +1;
            }else{
                e = m - 1;
            }
        }
        if(res != 0) break;
    }
    return String.valueOf(res);
        
    }
}
public int splitArray(int[] nums, int m){
    if(nums == null || nums.length == 0) return 0;
    long sum = 0;
    long max = 0;
    for(int num : nums){
        max = Math.max(max, num);
        sum += num;
    }
    if(sum % m != 0) return -1;
    while(max < sum){
        long mid = max + (sum - max)/2;
        if(isValid(nums, m, mid)){
            sum = mid;

        }else{
            max = mid + 1;
        }
    }
    return (int) max;

}

private boolean isValid(int[] nums, int m, int mid){
    int cnt = 1;
    int sum = 0;
    for(int i = 0; i < nums.length; i++){
        sum += nums[i];
        if(sum > mid){
            sum = nums[i];
            cnt++;
            if(cnt > m) return false;
        }
        
    }
    return true;
}


//This is difficult
 public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        if(m > n) return findMedianSortedArrays(nums2, nums1);
        int s = 0;
        int e = m;
        int maxleft;
        int minright;
        int i = Integer.MIN_VALUE, j = Integer.MIN_VALUE;
        while(s <= e){
            i = s + (e -s)/2;
            j = (m + n + 1)/2 - i;
            if(i > 0 && nums1[i - 1] > nums2[j]) {
                e = i - 1;
            }else if( i < m && nums2[j-1] > nums1[i]){
                s = i + 1;
            }else break;
        }
        if(i == 0) maxleft = nums2[j-1];
        else if( j == 0) maxleft = nums1[i - 1];
        else maxleft = Math.max(nums1[i-1] , nums2[j-1]);
        
        if((m + n) % 2 == 1) return maxleft;
        
        if(i == m) minright = nums2[j];
        else if( j == n) minright = nums1[i];
        else minright = Math.min(nums1[i] , nums2[j]);
        
        return (minright + maxleft)/2.0 ;
    }
//关于中位数： 如果奇数个： (n-1)/2， n/2 也可以 就是index。 
// 如果是偶数个： （n-1）／2 是前一个，n/2是后一个
 // 所以不分奇偶的时候可以用（n-1）／2 比较好来表示第一个index。如果分奇偶的话，就用两个表示。
public double findMedianSortedArrays(int[] nums1, int[] nums2){
    int m = nums1.length, n = nums2.length;
    int left = (m + n + 1)/2, right = (m + n + 2)/2;
    return (helper(nums1, nums2, right) + helper(nums1, nums2, left))/2.0;
}
private int helper(int[] n1, int[] n2, int index){
    if(index == 0) return -1;
    int m = n1.length; 
    int n = n2.length;
    if(m > n) return helper(n2, n1, index);
    // two base cases:
    if(m == 0) return n2[index - 1];
    if(k == 1) return Math.min(n1[0], n2[0]);
    int i = Math.min(m, index/2);
    int j = Math.min(n, index/2);
    if(n1[i -1 ] > n2[j -1]){
        helper(n1, Arrays.copyOfRange(n2, j, n), index - j);
    }else {
        helper(Arrays.copyOfRange(n1, i, m), n2, index - i);
    }
    
}


public int longestIncreasingPath(int[][] matrix){
    if(matrix.length == 0 || matrix[0].length == 0) return 0;
    int m = matrix.length;
    int n = matrix[0].length;
    int[][] dp = new int[m][n];
    int max = 1;
    for(int i = 0; i < m; i ++){
        for(int j = 0; j < n; j++){
            max = Math.max(max, helper(i, j, matrix, dp));
        }

    }
    return max;
}
private int helper(int i, int j, int[][] matrix, int[][] dp){
    if(dp[i][j] != 0) return dp[i][j];
    int[][] dir = new int[][]{{-1, 0}, {1, 0}, {0,1}, {0, -1}};
    int max = 1;
    for(int[] d : dir){
        int m = i + d[0];
        int n = j + d[1];
        if(m <0 || n < 0 || m >= matrix.length || n >= matrix[0].length ||matrix[m][n] <= matrix[i][j]) continue;
        int len = 1 + helper(m, n, matrix, dp); 
        max = Math.max(len, max);
    }
    dp[i][j] = max;
    return max;
}

public int longestIncreasingPath(int[][] matrix){
    if(matrix == null || matrix[0].length == 0) return 0;
    int m = matrix.length, n = matrix[0].length;
    int[][] dp = new int[m][n];
    int max = 1;
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[2] - a[2]);
    for(int i = 0; i < m ; i++){
        for(int j = 0; j < n; j++){
            pq.offer(new int[]{i, j, matrix[i][j]});
        }
    }
    int[][] dir = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    while(!pq.isEmpty()){
        int[] cur = pq.poll();
        int i = cur[0], j = cur[1];
        dp[i][j] = 1;
        for(int[] d : dir){
            int a = i + d[0];
            int b = j + d[1];
            if(a < 0 || b < 0 || a >= m || b >= n || matrix[a][b] <= matrix[i][j]) continue;
            int len = 1 + dp[a][b];
            dp[i][j] = Math.max(dp[i][j], len);
        }
        max = Math.max(max, dp[i][j]);
    }
    return max;
}
//this is a medium difficult problem:
  public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return res;
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[][] p = new boolean[m][n];
        boolean[][] at = new boolean[m][n];
        Queue<int[]> pq = new LinkedList<>();
        Queue<int[]> atq = new LinkedList<>();
        for(int i = 0; i < n ; i++){
            p[0][i] = true;
            at[m-1][i] = true;
            pq.offer(new int[]{0, i});
            atq.offer(new int[]{m-1, i});
        }
        for(int j = 0; j < m; j++){
            p[j][0] = true;
            at[j][n-1] = true;
            pq.offer(new int[]{j, 0});
            atq.offer(new int[]{j, n-1});          
        }
        bfs(matrix, p, pq);
        bfs(matrix, at, atq);
        for(int i = 0; i < m; i++){
            for(int j=0; j < n; j++){
                if(p[i][j] && at[i][j]){
                    res.add(new int[]{i, j});
                }
            }
        }
        return res;
        
    }
    private void bfs(int[][] matrix, boolean[][] p, Queue<int[]> q){
        int[][] dir = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        while(!q.isEmpty()){
            int[] cur = q.poll();
            int a = cur[0];
            int b = cur[1];
            for(int[] d : dir){
                int m = a + d[0];
                int n = b + d[1];
                if(m < 0 || n < 0 || m >= matrix.length || n >= matrix[0].length || p[m][n]||matrix[m][n] < matrix[a][b]) {
                    continue;
                }
                p[m][n] = true;
                q.offer(new int[]{m, n});
            }
        }
    }

    class Solution {
    public final int PAC = 1;
    public final int ATL = 2;
    public final int BOT = 3;
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return res;
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] status = new int[m][n];

        for(int i = 0; i < n ; i++){
            dfs(matrix, status,  0, i, PAC);
            dfs(matrix, status, m-1, i, ATL);
        }
        for(int j = 0; j < m; j++){
            dfs(matrix, status, j, 0, PAC);
            dfs(matrix, status, j, n-1, ATL);
        }
        for(int i = 0; i < m; i++){
            for(int j=0; j < n; j++){
                if(status[i][j] == BOT){
                    res.add(new int[]{i, j});
                }
            }
        }
        return res;
        
    }
    private void dfs(int[][] matrix, int[][] status, int i, int j, int sts){
        status[i][j] = status[i][j]|sts;
        int[][] dir = new int[][]{{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
        for(int[] d : dir){
            int p = i + d[0];
            int q = j + d[1];
            // Here is to compare bitmask with the input sts not the status[i][j];
            if(p >=0 && q >= 0 && p < matrix.length && q < matrix[0].length && matrix[p][q] >= matrix[i][j] && (status[p][q] & sts) == 0){
                dfs(matrix, status, p, q, sts);
            }
        }
        
        
    }
}

 public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
        if (k == 0) return W;
        PriorityQueue<int[]> pqC = new PriorityQueue<>((a,b) -> a[0] - b[0]);
        PriorityQueue<int[]> pqP = new PriorityQueue<>((a,b) -> b[1] - a[1]); 
        for(int i = 0; i < Profits.length; i++) {
            pqC.add(new int[]{Capital[i], Profits[i]});
        }
        for(int j = 0; j < k; j++){
            while(!pqC.isEmpty() && pqC.peek()[0] <= W){
                pqP.add(pqC.poll());   
            }
            if(pqP.isEmpty()) break;
            W += pqP.poll()[1];
        }
        return W;
        
    }

public List<Integer> countSmaller(int[] nums) {
    List<Integer> res = new ArrayList<>();
    List<Integer> sorted = new ArrayList<>();
    for(int i = nums.length -1 ; i >= 0; i--){
            int index = findLess(sorted, nums[i]);
            sorted.add(index, nums[i]);
            res.add(0, index);
    }
    return res;
}

private int findLess(List<Integer> sorted, int target) {
    if(sorted.isEmpty()) return 0;
    int s = 0, e = sorted.size() - 1;
    while(s <= e) {
        int mid = s + (e - s)/2 ;
        if(sorted.get(mid) >= target){
            e = mid - 1;
        }else if(sorted.get(mid) < target){
            s = mid + 1;
        } 
    }
    return s;
}
// binary search same effects  as the one above.
    // int s = 0, e = sorted.size();
    // while(s < e) {
    //     int mid = s + (e - s)/2 ;
    //     if(sorted.get(mid) >= target){
    //         e = mid;
    //     }else if(sorted.get(mid) < target){
    //         s = mid + 1;
    //     } 
    // }
    // return s;

class TreeNode{
    int smallCount;
    int val;
    TreeNode left;
    TreeNode right;
    public TreeNode(int count, int val){
        this.smallCount = count;
        this.val = val;
    }
}
public List<Integer> countSmaller(int[] nums){
    TreeNode root = null;
    int[] res = new Int[nums.length];
    if(nums == null || nums.length == 0) return Arrays.asList(res);
    for(int i =nums.length -1; i >= 0; i--) {
            root = insert(root, nums[i], res, i, 0);
    }
}

public TreeNode insert(TreeNode root, int val, int[] res, int index, int preSum){
    if(root == null){
        root = new TreeNode(0, val);
        res[index] = 0;
    }
    else if(root.val > val){
        root.smallCount++;
        root.left = insert(root.left, val, res, index, preSum);
    }else{
        root.right = insert(root.right, val, res, index, preSum + root.smallCount + (root.val < val? 1 : 0));
    }
    return root;
}


class Solution {
    public int removeBoxes(int[] boxes) {
        int n = boxes.length;
        int[][][] dp = new int[n][n][n];
        return helper(boxes, 0, n-1, 0, dp);
    }
    private int helper(int[] boxes, int i, int j, int k, int[][][] dp) {
        if(i > j) return 0;
        if(dp[i][j][k] > 0) return dp[i][j][k];
        while(i < j && boxes[i + 1] == boxes[i]){
            k++;
            i++;
        }
        int res = (k+1)*(k+1) + helper(boxes, i + 1, j, 0, dp);
        for(int m = i+2; m <= j; m++){
            if(boxes[i] == boxes[m]){
                res = Math.max(res, helper(boxes, i+1, m-1, 0, dp) + helper(boxes, m, j, (k+1), dp));
            }
        }
        dp[i][j][k] = res;
        return res;
    }
}

public int maxSumSubmatrix(int[][] matrix, int k) {
    if(matrix.length == 0) return 0;
    int m = matrix.length;
    int n = matrix[0].length;
    int result = Integer.MIN_VALUE;
    for(int left = 0; left < n; left ++){
         int[] sum = new int[m];
        for(int right = left; right < n; right ++){
            int sum = 0, maxSum = Integer.MIN_VALUE;
            for(int i = 0; i < m; i++){
                sum[i] += matrix[i][right];
                sum = Math.max(sum + sum[i], sum[i]);
                maxSum = Math.max(maxSum, sum);
            }
            if(maxSum <= k){
                result = Math.max(result, maxSum);
                continue;
            }
            TreeSet<Integer> set = new TreeSet<>();
            set.add(0); //this is for one dimensional first element equals to K: [3, 2] k = 3;
            int curSum = 0;
            for(int s: sum) {
                curSum += s;
                Integer c = set.ceiling(curSum - k);
                if(c != null) result = Math.max(result, curSum - c);
                set.add(curSum;)
            }


        }
    }
    return result;
}

public int maxEnvelopes(int[][] envelopes) {
    if(envelopes.length < 2) return envelopes.length;
    Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
    int[] dp = new int[envelopes.length];
    int size= 0;
    for(int[] cur :envelopes){
        int s = 0; 
        int e = envelopes.length;
        while(s < e){
            int mid = s + (e - s)/2;
            if(dp[mid] < cur[1]){
                 s = mid + 1;
            }else{
                e = mid;
            }
        }
        dp[left] = cur[1];
        if(left == size) size++;

    }
    return size;
}

public int lengthOfLIS(int[] nums) {
        if(nums.length < 2) return nums.length;
        int[] max = new int[nums.length];
        Arrays.fill(max, 1);
        int res = 1;
        for(int i = 0; i < nums.length; i++){
            for(int j = 0; j < i; j++){
                if(nums[i] > nums[j]){
                    max[i] = Math.max(max[i], max[j] + 1);
                }
            
            }
            res = Math.max(res, max[i]);
        }
      return res;  
    }

 public int lengthOfLIS(int[] nums) {
        if(nums.length < 2) return nums.length;
        List<Integer> list = new ArrayList<>();
        for(int num : nums){
            if(list.size() == 0 || num > list.get(list.size() - 1)){
                list.add(num);
            }else{
                int s = 0;
                int e = list.size() - 1;
                while(s < e){
                    int mid = s + (e - s)/2;
                    if(list.get(mid) < num){
                        s = mid + 1;
                    } else{
                        e = mid;
                    }
                }
                list.set(s, num);
            }
        }
        return list.size();
    }


//frog jump: one iteration, one recursion


    