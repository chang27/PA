import java.util.*;

/**
 * Created by Chang on 11/4/17.
 */
public class canReach {
   // static String res = "";
    public static String canReach(int x1, int y1, int x2, int y2) {
        StringBuilder sb = new StringBuilder();
        sb.append(x1).append(y1);
        if(x1 == x2 && y1 == y2){
            return sb.toString();
        }
        //String res = null;

       if(dfs(x1, y1, x2, y2, sb)) return sb.toString();
        return null;
    }
    private static boolean dfs(int x1, int y1, int x2, int y2, StringBuilder sb){
        if(x1 > x2 || y1 > y2) return false;

        if(x1 == x2 && y1 == y2) {
            System.out.println(sb.toString());
//            if(res.length() == 0 || res.length() < sb.toString().length()){
//                res = sb.toString();
//            }
       //     System.out.println(res);
            return true;
        }
       if(dfs(x1, x1 + y1, x2, y2, sb.append(x1).append(x1 + y1))){
           return true;
        }
        sb.delete(sb.length()- 2, sb.length());
       if(dfs(x1 + y1, y1, x2, y2, sb.append(x1+y1).append(y1))){
           return true;

        }
        return false;
//        System.out.println("run through here 1 ");
//     //   System.out.println(x1 + " "+  y1);
//        System.out.println(sb.toString());
//        sb.delete(sb.length()- 2, sb.length());
//        dfs(x1 + y1, y1, x2, y2, sb.append(x1+y1).append(y1));
//        System.out.println("run through here 2 ");
    }

    int[][] dir = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
//    int[][] trans = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    public int num(int[][] grid) {
        Set<String> set = new HashSet<>();
        int m = grid.length;
        int n = grid[0].length;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++) {
                if(grid[i][j] == 1){
                    List<int[]> cells = new ArrayList<>();
                    dfs(grid, i, j, 0, 0, cells);
                    String key = norm(cells);

                }
            }
        }
        return set.size();
    }
    private void dfs(int[][] grid, int i, int j, int oi, int oj, List<int[]> list){
        grid[i][j] = 0;
        list.add(new int[]{oi, oj});
        for(int k = 0; k < 4; k++){
            int x = i + dir[k][0];
            int y = j + dir[k][1];
            if(x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] == 1){
                dfs(grid, x, y, oi + dir[k][0], oj + dir[k][1], list);
            }
        }
    }
    int[][] trans = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    private String norm(List<int[]> cells){
        List<String> forms = new ArrayList<>();
        for(int[] ts : trans){
            List<int[]> list1 = new ArrayList<>();
            List<int[]> list2 = new ArrayList<>();
            for(int[] c : cells){
                list1.add(new int[]{c[0]*ts[0], c[1] * ts[1]});
                list2.add(new int[]{c[1]*ts[0], c[0] * ts[1]});
            }
            forms.add(getKey(list1));
            forms.add(getKey(list2));
        }
        Collections.sort(forms);
        return forms.get(0);
    }

    private String getKey(List<int[]> list){
        Collections.sort(list, (a,b)-> a[0] == b[0]? a[1] - b[1] : a[0] - b[0]);
        StringBuilder sb = new StringBuilder();
        int ox = list.get(0)[0];
        int oy = list.get(0)[1];
        for(int[] c : list){
            sb.append((c[0] - ox) + "" + (c[1] - oy));
        }
        return sb.toString();
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> map = new HashMap<>();
        Map<String, String> parent = new HashMap<>();
        for(List<String> list : accounts){
            map.put(list.get(1), list.get(0));
            for(int i = 1; i < list.size(); i++) {
                String p = find(list.get(1), parent);
                parent.put(find(list.get(i), parent), p);
            }
        }
        Map<String, TreeSet<String>> union = new HashMap<>();
        for(String s : parent.keySet()){
            if(! union.containsKey(parent.get(s))){
                union.put(parent.get(s), new TreeSet<>());
            }
            union.get(parent.get(s)).add(s);
        }
        List<List<String>> res = new ArrayList<>();
        for(String s : union.keySet()){
            List<String> sub = new ArrayList<>(union.get(s));
            sub.add(0, map.get(s));
            res.add(sub);
        }

        return res;
    }

    private String find(String s, Map<String, String> map){
        if(! map.containsKey(s)){
            map.put(s, s);
        }
        while(! s.equals(map.get(s))){
            s = map.get(s);
        }
        return s;
    }

    public static void printSeq(int a, int b){
        if(a > b) return;
        int i = a;
        boolean increase = true;
       // System.out.println(i);
        while(i >= a && i <= b){
            System.out.println(i);
            if(i == a && !increase) break;
            else if(i == b || !increase){
                i--;
                increase = false;
            }
            else if(increase) i++;
        }
    }


//    public static void printSeq1(int a, int b){
//        if(a > b){
//            return;
//        }
//        printRecursive(b);
//        printRecursive(a);
//
//    }

    public static void printRecursive(int a, int b){
        if(a > b) return;
        System.out.print(a);
        if(a < b) System.out.print(" ");
        printRecursive(a+1, b);
        if(a < b) {
            System.out.print(" " + a);
        }
    }

    public static void main(String[] args) {
        printRecursive(3, 7);

    }




}

