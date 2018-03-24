import java.util.*;

/**
 * Created by Chang on 1/30/18.
 */
public class printRecursive {
    public static void printRecursive(int a, int b){
        if(a > b) return;
        System.out.print(a);
        if(a < b) System.out.print(" ");
        printRecursive(a+1, b);
        if(a < b) {
            System.out.print(" " + a);
        }
    }

    public static boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String, List<String>> map = new HashMap<>();
        for(String s : allowed) {
            String key = s.substring(0, 2);
            if(! map.containsKey(key)){
                map.put(key, new ArrayList<>());
            }
            map.get(key).add(s.substring(2));
        }
        System.out.println(map.size());
        return help(bottom, map);
    }
    private static boolean help(String bottom, Map<String, List<String>> map){
        if(bottom.length() == 1) return true;
        for(int i = 0; i < bottom.length() - 1; i++){
            if(! map.containsKey(bottom.substring(i, i + 2))) return false;
        }
        List<String> res = new ArrayList<>();
        dfs(res, map, bottom, 0, new StringBuilder());
        System.out.println(res.size());
        for(String s : res){
            if(help(s, map)) return true;
        }
        return false;
    }
    private static void dfs(List<String> res, Map<String, List<String>> map, String s, int idx, StringBuilder sb){
        if(idx == s.length()-1){
            res.add(sb.toString());
            return;
        }
        for(String sub : map.get(s.substring(idx, idx + 2))){
            sb.append(sub);
            dfs(res, map, s, idx +1, sb);
            sb.deleteCharAt(sb.length()-1);
        }

    }


    public static int calculate(String s) {
        if (s == null || s.length() == 0) return 0;
        int n = s.length();
        char[] array = s.toCharArray();
        Stack<Integer> stack1 = new Stack<>();
        Stack<Character> ops = new Stack<>();
        for(int i = 0; i < n; i++){
            if(array[i] == ' ') continue;
            if(Character.isDigit(array[i])){
                int num = array[i] - '0';
                while(i + 1 < n && Character.isDigit(array[i+1])){
                    num = num * 10 + (array[i+1] - '0');
                    i++;
                }
                stack1.push(num);

            }else if(array[i] == '('){
                ops.push(array[i]);
            }else if(array[i] == ')'){
                while(ops.peek() != '('){
                    stack1.push(cal(stack1.pop(), stack1.pop(), ops.pop()));
                }
                ops.pop();
            }else if(array[i] == '*' || array[i] == '/' || array[i] == '+' || array[i] == '-'){
                while(! ops.empty() && prior(ops.peek(), array[i])){
                    stack1.push(cal(stack1.pop(), stack1.pop(), ops.pop()));
                }
                ops.push(array[i]);
            }
        }
        while(!ops.empty()){
            stack1.push(cal(stack1.pop(), stack1.pop(), ops.pop()));
        }
        return stack1.pop();
    }
    // c1 is higher than c2:
    private static boolean prior(char c1, char c2){
        return ((c1 == '*' || c1 == '/') && (c2 =='+'|| c2 =='-'));
    }

    private static int cal(int b, int a, Character c){
        switch(c){
            case '+': return a + b;
            case '-': return a - b;
            case '*': return a * b;
            case '/': return a/b;
        }
        return 0;
    }


    public int[][] candyCrush(int[][] board) {
        int R = board.length;
        int C = board[0].length;
        boolean finish = true;
        for(int i = 0; i < R; i++){
            for(int j = 0; j < C - 2; j++){
                if(Math.abs(board[i][j])!= 0 && Math.abs(board[i][j]) == Math.abs(board[i][j+1]) && Math.abs(board[i][j]) == Math.abs(board[i][j+2])){
                    int v = Math.abs(board[i][j]);
                    board[i][j] = board[i][j+1] = board[i][j+2] = -v;
                    finish = false;
                }
            }
        }
        for(int i = 0; i < R - 2; i++){
            for(int j = 0; j < C; j++){
                int v = Math.abs(board[i][j]);
                if(v!= 0 && v == Math.abs(board[i+1][j]) && v == Math.abs(board[i+2][j])){
                    board[i][j] = board[i+1][j] = board[i+2][j] = -v;
                    finish = false;
                }
            }
        }
        for(int c = 0; c < C; c++){
            int s = R - 1;
            for(int i = R-1; i >= 0; i--){
                if(board[i][c] > 0){
                    board[s--][c] = board[i][c];
                }
            }
            while(s >= 0){
                board[s--][c] = 0;
            }
        }
        return finish? board : candyCrush(board);


    }

    public int networkDelayTime(int[][] times, int N, int K) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for(int[] edge : times){
            if(!graph.containsKey(edge[0])){
                graph.put(edge[0], new ArrayList<>());
            }
            graph.get(edge[0]).add(new int[]{edge[1], edge[2]});
        }
        PriorityQueue<int[]> qp = new PriorityQueue<int[]>((a, b) -> a[0]- b[0]);
        qp.offer(new int[]{0, K});
        //Map<Integer, Integer> dist = new HashMap<>();
        int[] dist = new int[N+1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        while(!qp.isEmpty()){
            int[] cur = qp.poll();
            int d = cur[0];
            int node = cur[1];
            //    if(!dist.containsKey(node)){
            dist[node] = d;
            //   dist.put(node, d);
            if(graph.containsKey(node)){
                for(int[] edge : graph.get(node)){
                    int nei = edge[0];
                    int dis = edge[1];
                    if(dist[nei] > dist[node] + d){
                        qp.offer(new int[]{dis + d, nei});
                    }
                }
            }
            //        }
            //     }
            //}

        }
        int max = 0;
        for(int i = 1; i <= N; i++){
            max = Math.max(max, dist[i]);
        }
        return max == Integer.MAX_VALUE? -1 : max;
    }

    static Set<String> seen;
   static  StringBuilder ans;

    public static String crackSafe(int n, int k) {
        if (n == 1 && k == 1) return "0";
        seen = new HashSet();
        ans = new StringBuilder();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n-1; ++i)
            sb.append("0");
        String start = sb.toString();

        dfs(start, k);
        ans.append(start);
        return new String(ans);
    }

    public static void dfs(String node, int k) {
        for (int x = 0; x < k; ++x) {
            String nei = node + x;
            // 00, 01, 10, 11;
            if (!seen.contains(nei)) {
                System.out.println("here" + nei);
                seen.add(nei);
                dfs(nei.substring(1), k);
                System.out.println("ans" + ans.toString());
                ans.append(x);
            }
        }
    }

    public int containVirus(int[][] grid){
        int m = grid.length;
        int n= grid[0].length;
        int total = 0;
        while(true){
            Set<Integer> visited = new HashSet<>();
            List<Integer> virus = new ArrayList<>();
            List<Set<Integer>> next = new ArrayList<>();
            int idx = -1;
            int num = -1;
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    int key = i * n + j;
                    if(grid[i][j] != 1 || visited.contains(key)) continue;
                    List<Integer> cur = new ArrayList<>();
                    Set<Integer> nn = new HashSet<>();
                    int wall = getArea(j, i, m, n, grid, visited, cur, nn);
                    if(nn.isEmpty()){
                        continue;
                    }
                    if(next.isEmpty()|| nn.size() > next.get(idx).size()){
                        virus = new ArrayList<>(cur);
                        cur.clear();
                        idx = next.size();
                        num = wall;
                    }
                    next.add(new HashSet<>(nn));
                }
            }
            if(next.isEmpty()){
                break;
            }
            total += num;
            for(int i = 0; i < next.size(); i++){
                if(i == idx){
                    for(int key : virus){
                        int y = key/n;
                        int x = key % n;
                        grid[y][x] =  2;
                    }
                }else{
                    for(int k : next.get(i)){
                        int y = k/n;
                        int x = k%n;
                        grid[y][x] = 1;

                    }
                }
            }

        }
        return total;
    }

    private int getArea(int x, int y, int m, int n, int[][] grid, Set<Integer> visited, List<Integer> cur, Set<Integer> next){
        int wall = 0;
        int key = y * n + x;
        if(x < 0 || x >= n || y < 0 || y >= m || grid[y][x] == 2) return wall ;
        if(grid[y][x] == 0){
            wall += 1;
            next.add(key);
            return wall;
        }
        if(visited.contains(key)){
            return wall;
        }
        visited.add(key);
        cur.add(key);
        int[] dir = { -1, 0, 1, 0, -1 };
        for(int i =0; i < 4; i++){
            wall += getArea(x + dir[i], y + dir[i+1], m, n, grid, visited, cur, next);
        }
        return wall;

    }

    public static int minStickers(String[] stickers, String target) {
        int m = stickers.length;
        int[][] dic = new int[m][26];
        for(int i = 0; i < m; i++){
            String s = stickers[i];
            for(char c : s.toCharArray()){
                dic[i][c - 'a']++;
            }
        }

        Map<String, Integer> dp = new HashMap<>();
        dp.put("", 0);
        return dfs(dic, dp, target);

    }
    private static int dfs(int[][] dic, Map<String, Integer> map, String target){
        // if(target.length() == 0) return 0;
        if(map.containsKey(target)) return map.get(target);
        int[] tar = new int[26];
        for(char c : target.toCharArray()){
            tar[c - 'a']++;
        }
        int ans = Integer.MAX_VALUE;
        int n = dic.length;
        for(int i = 0; i < n; i++){
            if(dic[i][target.charAt(0) - 'a'] == 0) continue;
            StringBuilder sb = new StringBuilder();
            for(int j = 0; j < 26; j++){
                if(tar[j] > 0){
                    for(int k = 0; k < Math.max(tar[j] - dic[i][j], 0); k++){
                        sb.append((char)('a' + j));
                    }
                }
            }

            String next = sb.toString();
            System.out.println(next);
            int val = dfs(dic, map,next);
            if(val != -1) ans = Math.min(ans, val + 1);
        }
        //ans == Integer.MAX_VALUE? -1 : ans;
        map.put(target, ans == Integer.MAX_VALUE? -1 : ans);
        return map.get(target);
    }


    public static void main(String[] args) {
//       String s =  "ABC";
//        String a[] = new String[]{"ABD","BCE","DEF","FFF"};
//       List<String> list = Arrays.asList(a)    ;  //   ["ABD","BCE","DEF","FFF"]
//        System.out.println(pyramidTransition(s, list));
//        String s = "2*(5+5*2)/3+(6/2+8)";
//        System.out.println(calculate(s));

//        String res = crackSafe(2, 2);
//        System.out.println(res);
        String[] list = {"with","example","science"};
        String tar = "thehat";
        int s = minStickers(list, tar);

    }
}
