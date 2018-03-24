package edu.uci.ics.textdb.exp.dictionarymatcher;

import java.util.*;

/**
 * Created by Chang on 9/21/17.
 */
public class findKthNumber {
    public int findKthNumber(int n, int k){
        int cur = 1;
        k = k - 1;
        while(k > 0){
            int step = calSteps(n, cur, cur+ 1);
            if (step <= k){
                k -= step;
                cur++;
            }else{
               cur *= 10;
                k--;
            }
        }
        return cur;
    }
    private int calSteps(int n, long n1, long n2){
        int step = 0;
        while(n1 <= n){ // there is <=
            step += Math.min(n+1, n2) - n1;
            n1 *= 10;// n1 and n2 could be overflow
            n2 *= 10;
        }
        return step;
    }

    public List<Integer> printOrder(int n) {
        List<Integer> res = new ArrayList<>();
        int cur = 1;
        for(int i = 0; i < n; i++){
            res.add(cur);
            if(cur * 10 <= n){
                cur *= 10;
            }else{
                if(cur + 1 <= n && cur%10 != 9){
                    cur++;
                }else{
                    cur /= 10;
                    while(cur %10 == 9){
                        cur /= 10;
                    }
                    cur++;
                }

            }
        }
        return res;

    }
    public class compa implements Comparator<Integer>{

//        public int compare(String s1, String s2) {
//
//            return s1.compareTo(s2);
//        }
        public int compare(Integer a, Integer b){
            return a - b;
        }

    }
    public static void main(String[] args) {
        findKthNumber fk = new findKthNumber();

        List<String> list = new ArrayList<>();
        list.sort((a,b) -> a.compareTo(b));
        Collections.sort(list, (a,b) -> a.compareTo(b));
        TreeSet<Integer> set = new TreeSet<>((a, b) -> a - b);
       // Set<Integer> set1 = new HashSet<compa>();

        // map.putIfAbsent("abc", new ArrayList<>((a,b) -> a.compareTo(b)));
        List<Integer> res = fk.printOrder(13);
        System.out.println(res.toString());
    }
}
