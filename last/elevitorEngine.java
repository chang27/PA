package edu.uci.ics.textdb.exp.dictionarymatcher;

/**
 * Created by Chang on 11/20/17.
 */
public class elevitorEngine extends Thread {

    private boolean stopFlag = false;
    public void startEngine(){
        stopFlag = false;
        this.start();

    }
    public void stopEngine(){
        stopFlag = true;
    }

    public void run(){
        while(true){
            if(stopFlag){
                break;
            }
            try {
                Thread.sleep(1000);
            }catch(InterruptedException e){
                e.printStackTrace();
            }
        }
    }
}
