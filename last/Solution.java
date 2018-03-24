package edu.uci.ics.textdb.exp.dictionarymatcher;

/**
 * Created by Chang on 9/19/17.
 */
import java.util.*;
public class Solution {

    int[][] dir = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};

    public int cutOffTree(List<List<Integer>> forest) {
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[2] - b[2];
            }
        });
        for (int i = 0; i < forest.size(); i++) {
            for (int j = 0; j < forest.get(0).size(); j++) {
                pq.offer(new int[]{i, j, forest.get(i).get(j)});
            }
        }
        int sum = 0;
        int[] start = new int[2];
        while (!pq.isEmpty()) {
            int[] target = pq.poll();
            int step = bfs(forest, start, target);
            if (step == -1) return -1;

            sum += step;
            start = target;

        }
        return sum;
    }

    private int bfs(List<List<Integer>> forest, int[] start, int[] end) {
        Queue<int[]> q = new LinkedList<>();
        q.offer(start);
        int step = 0;
        while (!q.isEmpty()) {
            for (int i = q.size() - 1; i >= 0; i--) {
                int[] s = q.poll();
                if (s[0] == end[0] && s[1] == end[1]) return step;
                for (int[] d : dir) {
                    int j = s[0] + d[0];
                    int k = s[1] + d[1];
                    if (j < 0 || k < 0 || j > forest.size() || k > forest.get(0).size() || forest.get(j).get(k) == 0)
                        continue;
                    q.offer(new int[]{j, k});
                }
            }
            step++;
        }
        return -1;
    }


    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int n = nums1.length;
        int m = nums2.length;
        int[] res = new int[k];
        for (int i = Math.max(0, k - m); i <= k && i <= n; i++) {
            int[] part1 = get(nums1, i); // i indicates the range;
            int[] part2 = get(nums2, k - i);
            int[] cadi = merge(part1, part2, k);
            if (greater(cadi, res, 0, 0)) {
                res = cadi;
            }
        }
        return res;

    }

    private int[] merge(int[] n1, int[] n2, int size) {
        int[] res = new int[size];
        int m = 0;
        int n = 0;
        for (int i = 0; i < size; i++) {
            if (greater(n1, n2, m, n)) {
                res[i] = n1[m++];

            } else res[i] = n2[n++];
        }
        return res;
    }

    private boolean greater(int[] n1, int[] n2, int i, int j) {
        while (i < n1.length && j < n2.length && n1[i] == n2[j]) {
            i++;
            j++;
        }
        return j == n2.length || i < n1.length && n1[i] > n2[j];
    }

    private int[] get(int[] nums, int size) {
        int[] res = new int[size];
        int last = -1;
        for (int i = 0; i < size; i++) {
            for (int j = last + 1; j + size - i - 1 < nums.length; j++) {
                if (nums[j] > res[i]) {
                    res[i] = nums[j];
                    last = j;
                }
            }
        }
        return res;
    }

    public void solve(char[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0) return;
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            if (board[i][0] == '0') {
                check(board, i, 0, m, n);
            }
            if (n - 1 > 0 && board[i][n - 1] == '0') {
                check(board, i, n - 1, m, n);
            }
        }

        for (int j = 0; j < n; j++) {
            if (board[0][j] == '0') {
                check(board, 0, j, m, n);
            }
            if (m - 1 > 0 && board[m - 1][j] == '0') {
                check(board, m - 1, j, m, n);
            }
        }


        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == '1') {
                    board[i][j] = '0';
                } else if (board[i][j] == '0') {
                    board[i][j] = 'X';
                }
            }
        }

    }

    private void check(char[][] board, int i, int j, int m, int n) {
        board[i][j] = '1';
        if (i - 1 >= 0 && board[i - 1][j] == '0') {
            check(board, i - 1, j, m, n);
        }
        if (j - 1 >= 0 && board[i][j - 1] == '0') {
            check(board, i, j - 1, m, n);
        }
        if (i + 1 < m && board[i + 1][j] == '0') {
            check(board, i + 1, j, m, n);
        }
        if (j + 1 < n && board[i][j + 1] == '0') {
            check(board, i, j + 1, m, n);
        }
    }

    public boolean canFinish(int n, int[][] pre) {
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(new ArrayList<>());
        }
        for (int[] p : pre) {
            list.get(p[1]).add(p[0]);
        }
        int[] visited = new int[n];
        for (int i = 0; i < n; i++) {
            if (!list.get(i).isEmpty() && dfs(list, visited, i)) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs(List<List<Integer>> list, int[] visited, int i) {
        if (visited[i] == 2) return false;
        if (visited[i] == 1) return true;
        visited[i] = 1;
        for (Integer j : list.get(i)) {
            if (visited[j] == 0 && dfs(list, visited, j)) {
                return true;
            }
        }
        visited[i] = 2;
        return false;

    }

    public List<Integer> topKFrequent(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || k <= 0) return res;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer>[] freq = new List[nums.length + 1];
        ;
        for (Map.Entry<Integer, Integer> en : map.entrySet()) {
            int f = en.getValue();
            if (freq[f] == null) {
                freq[f] = new ArrayList<>();
            }
            freq[f].add(en.getKey());
        }
        for (int j = nums.length; j >= 0; j--) {
            List<Integer> l = freq[j];
            for (int i = 0; i < l.size() && res.size() < k; i++) {
                res.add(l.get(i));

            }
            if (res.size() == k) break;
        }
        return res;
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (a, b) -> a[0] == b[0] ? a[1] - b[1] : b[0] - a[0]);
        List<int[]> res = new ArrayList<>();
        for (int[] peo : people) {
            res.add(peo[1], peo);
        }
        return res.toArray(new int[res.size()][2]);
    }

    public String optimalDivision(int[] nums) {
        StringBuilder sb = new StringBuilder("");
        if (nums.length == 0) return sb.toString();
        if (nums.length == 1) {
            sb.append(nums[0]);
            return sb.toString();
        }
        if (nums.length == 2) {
            sb.append(nums[0]).append("/").append(nums[1]);
            return sb.toString();
        }
        sb.append(nums[0]).append("/(").append(nums[1]);
        for (int i = 2; i < nums.length; i++) {
            sb.append("/").append(nums[i]);
        }
        sb.append(")");
        return sb.toString();
    }


    public String convert(String s, int numRows) {
        if (numRows < 2 || numRows >= s.length()) return s;
        int len = s.length();
        StringBuilder[] sb = new StringBuilder[numRows];
        for (int i = 0, row = -1, d = 1; i < len; ) {
            if (d + row < 0 || d + row >= numRows) {
                d = -d;
            }
            sb[row].append(s.charAt(i++));


        }
        return String.join("", sb);
    }

    public static List<String> subs(String s) {
        Map<String, List<String>> map = new HashMap<>();
        map.put("a", new ArrayList<>());
        map.get("a").add("b");
        map.get("a").add("c");
        map.get("a").add("d");

     //   System.out.println(map.values());
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        helper(s, map, res, 0, sb);
        return res;
    }

    private static void helper(String s, Map<String, List<String>> map, List<String> list, int idx, StringBuilder sb) {
        if (idx == s.length()) {
            list.add(sb.toString());
        } else {
            if (!map.containsKey(String.valueOf(s.charAt(idx)))) {
                sb.append(s.charAt(idx));
                helper(s, map, list, idx + 1, sb);
                sb.setLength(sb.length() - 1);
            } else {
                for (String a : map.get(String.valueOf(s.charAt(idx)))) {
                    sb.append(a);
                    helper(s, map, list, idx + 1, sb);
                    sb.setLength(sb.length() - 1);
                }
            }
        }
    }

    public int[] singleNumber(int[] nums) {
        List<Integer> res = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            int idx = Math.abs(nums[i]) - 1;
            nums[idx] = -nums[idx];
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < 0) {
                res.add(i + 1);
                nums[i] = -nums[i];
            }

        }
        return res.stream().mapToInt(i -> i).toArray();
    }
    public static int minMoves2(int[] nums) {
        int res = 0;
        int midx = findMid(nums, nums.length/2, 0, nums.length - 1);
        System.out.println(midx);
        for(int num : nums){
            res += Math.abs(num - midx);
        }
        return res;
    }

    private static int findMid(int[] nums, int k, int s, int e){
        int pivot = nums[(e+s)/2];
        int i = s, j = e;
        while(i <= j){
            while(i <= j && nums[i] < pivot) i++;
            while(j >= i && nums[j] >= pivot) j--;
            if(i >= j){
                break;
            }
            swap(nums, i, j);

        }
        swap(nums, i, e/2);
        if(i == k){
            return nums[i];
        }else if( i > k){
            return findMid(nums, k, s,i-1);
        }else return findMid(nums, k,i+1, e);

    }
    private static void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public static int removeString(String s, String t) {
        if(s == null || s.length() == 0) return 0;
        int num = recursion(s, t, 0);
        return num;
    }

    private static int recursion(String s, String t, int i){
        if(s == null || s.length() == 0 || s.indexOf(t) == -1) return 0;
        int res = 0;
        List<Integer> list = new ArrayList<>();
        for(int j = i; j <s.length();){
            int idx = s.indexOf(t, j);
            if(idx != -1) {
                list.add(idx);
                j = idx + 1;
            }else{
                break;
            }
        }
        for(Integer ii : list) {
            int num = 1 + recursion(s.substring(0, ii) + s.substring(ii + t.length()), t, 0);
            res = Math.max(res, num);
        }
        return res;
    }


    public static void main(String[] args) {
        String s = "ababaababa";
        String t = "aba";
        int rr = removeString(s, t);
        System.out.println(rr);
//        List<String> res = subs(s);
//        System.out.println(res.toString());
//        int[] num = new int[]{2, 1};
//        int rs = findMid(num, 1, 0, 1);
//        System.out.println(rs);

    }

}
