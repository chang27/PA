//import java.util.*;
//
///**
// * Created by Chang on 12/29/17.
// */
//public final class Test4 {
//    class Inner{
//        void test(){
//            if(Test4.this.flag){
//                sample();
//            }
//        }
//
//    }
//    private boolean flag = true;
//    public void sample(){
//        System.out.println("sample");
//    }
//    public Test4(){
//        (new Inner()).test();
//    }
//
//    public static int divide(int a, int b){
//        int c = -1;
//        try{
//            c = a / b;
//
//        }catch(Exception e){
//            System.out.println("Exception");
//        }
//        finally{
//            System.out.println("Finally");
//        }
//        return c;
//    }
//
//    private boolean isVowel(char c){
//        return false;
//    }
//
//
//    ArrayList<Iterator> list;
//////   public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
////        list = new ArrayList<>();
////        if(!v1.isEmpty()) list.add(v1.iterator());
////        if(!v2.isEmpty()) list.add(v2.iterator());
////    }
//
//    public int next() {
//   //     Iterator iter = list.remove();
////
//// //       Integer res = iter.next();
////        if(iter.hasNext()){
////            list.add(iter);
////        }
//// //       return res;
////    }
//
// //   public boolean hasNext() {
////        return !list.isEmpty();
////    }
////    public static void main(String args []) {
////     //   new Test4();
////       // int res = divide(4, 0);
////        String s = "abcdfedfd";
////        char[] array = s.toCharArray();
////        String convert = String.copyValueOf(array);
////        //List<Integer> list1 = new ArrayList<>();
////
////
////        //System.out.println(s.substring(10));
////    }
////
////}
/////