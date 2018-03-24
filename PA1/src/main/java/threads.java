/**
 * Created by Chang on 2/12/18.
 */
public class threads extends Thread {
    private static Object lock = new Object();
    protected static int current = 1;
    private int max;
    private boolean div3, div5;
    private String toPrint;
    private int id;

    public threads(int id, boolean div3, boolean div5, int max, String toPrint) {
        this.div3 = div3;
        this.div5 = div5;
        this.max = max;
        this.toPrint = toPrint;
        this.id = id;
    }
    public void print(){
        System.out.println(toPrint);
    }

    public void run(){
        System.out.println("which thread" + id);
        while(true){
            synchronized (lock){
                if(current > max){
                    return;
                }
                System.out.println(current);
                if((current % 3 == 0 ) ==  div3  && (current % 5 == 0) == div5){
                    print();
                    current++;
                }
            }
        }
    }

    public static void main(String[] args){
        int n = 2;
        Thread[] th = {new threads(0, true, true, n, "fb"), new threads(1,true, false, n, "f"),
        new threads(2,false, true, n, "b"), new threads(3,false, false, n, "none")};
        for(int i = 0; i < th.length; i++){
            System.out.println(i);
           th[i].start();
        }
    }

}
