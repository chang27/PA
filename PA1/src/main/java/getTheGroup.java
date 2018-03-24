import java.util.*;

/**
 * Created by Chang on 11/4/17.
 */
public class getTheGroup {
    public static String friends = "Friends";
    public static String union = "Unions";
    public static Map<Integer, List<Integer>> friendCycle;

    public static void  getTheGroup(int n){
        friendCycle = new HashMap<>();
        for(int i = 0; i < n; i++){
            friendCycle.put(i, new ArrayList<>());
        }
    }
    public static int[] finalResult(int n, int s, String[] queryTypeArray, int s1, int[] students1, int s2, int[] students2){
        getTheGroup(n);
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < s; i++){
             int r =   getTheGroups(n, queryTypeArray[i], students1[i], students2[i]);
             if(r > 0) {
                 res.add(r);
             }
        }
        return res.stream().mapToInt(i -> i).toArray();
    }
    public static int getTheGroups(int n, String queryType, int students1, int students2) {

        if(queryType.equals(friends)){
            friends(students1, students2);
            return 0;
        }else{
            int res = count(students1, students2);

            return res;
        }
    }

    private static void friends(int i, int j){
        if(friendCycle.get(i).contains(j)){
            return;
        }
        friendCycle.get(i).add(j);
        friendCycle.get(j).add(i);
    }
    private static int count(int i, int j) {
        Set<Integer> set = new HashSet<>();
        if(friendCycle.get(i).contains(j)){
             getCount(i, set);

        }else{
            getCount(i, set);
            getCount(j, set);
        }
        return set.size();
    }
    private static void getCount(int i, Set<Integer> set){
        if(!friendCycle.containsKey(i)){
            return;
        }
        Queue<Integer> adj = new LinkedList<>();
        adj.add(i);
        set.add(i);
        while(!adj.isEmpty()) {
            int k = adj.poll();

            for (int j : friendCycle.get(k)) {
                if (set.add(j)) {
                    adj.add(j);
                }
            }
        }

        //set.add(i);
    }

    public static void main(String[] args) {
        getTheGroup(5);
        int res = getTheGroups(5, friends, 1, 2);
        int res1 = getTheGroups(5, union, 2, 3);
      //  int res2 = getTheGroups(5, friends, 1, 0);
        System.out.println( res1);


    }

}
