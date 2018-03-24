package edu.uci.ics.textdb.exp.dictionarymatcher;
import java.util.*;
/**
 * Created by Chang on 11/18/17.
 */
public class rangeQuery {

    class Interval{
        int start;
        int end;
        public Interval(int start, int end){
            this.start = start;
            this.end = end;
        }
    }
    List<Interval> list;
//    public String toString(List<Interval> list){
//        for()
//    }
    public rangeQuery() {
        list = new ArrayList<>();
    }

    public void addRange(int left, int right) {
        if(list.isEmpty() || list.get(list.size() - 1).end < left){
            list.add(new Interval(left, right)); // add this new interval at the end of the list;
            return;
        }

        for(int i = 0; i <= list.size(); i++){
//            if(list.get(i).end < left){
//                continue;
//            }
            if(i == list.size() || list.get(i).start > right){
                list.add(i, new Interval(left, right));
                return;
            }else if(list.get(i).end < left) {
                continue;
            }else{
                left = Math.min(list.get(i).start, left);
                right = Math.max(list.get(i).end, right);

//                if(i == list.size() - 1){
//                    list.remove(i);
//                    list.add(new Interval(left, right));
//                    return;
//                }
                list.remove(i);
                i--;
               // return;
            }
        }

    }

    public boolean queryRange(int left, int right) {  //binary Search:
        int n = list.size();
        int s = 0;
        int e = n-1;
        while(s <= e){
            int mid = s + (e - s)/2;
            if(list.get(mid).start >=right){
                e = mid -1;
            }else if(list.get(mid).end <= left){
                s = mid + 1;
            }else{
                if(list.get(mid).start<= left && list.get(mid).end >= right){
                    return true;
                }else {
                    return false;
                }
            }
        }
        return false;

    }

    public void removeRange(int left, int right) {
        int n = list.size();
        if(n == 0) return;
        for(int i = 0; i < list.size(); i++){
            if(list.get(i).end <= left) continue;
            if(list.get(i).start > right) return;

            if(list.get(i).start >= left){
                if(list.get(i).end <= right){
                    list.remove(i);
                    i--;
                   // continue;
                }
                else{
                    list.get(i).start = right;
                    return;
                }
            }
            else if(list.get(i).start < left){
                int tmp = list.get(i).end;
                list.get(i).end = left;
                if(tmp > right){
                    i++;
                    list.add(i, new Interval(right, tmp));
                }

            }
        }
    }

    public static void main(String[] args){
        rangeQuery q = new rangeQuery();
        q.addRange(6, 8);
       //q.queryRange(3, 4);
        q.removeRange(7, 8);
        q.removeRange(8,9);
        System.out.println(q.list.toString());
        q.addRange(8, 9);
        q.removeRange(1, 3);
        q.addRange(1, 8);
       // q.addRange(4, 8);
       System.out.println(q.queryRange(2, 4));
      //  q.removeRange(4, 9);
        System.out.println(q.queryRange(2, 9));
        System.out.println(q.queryRange(4, 6));

    }
}
