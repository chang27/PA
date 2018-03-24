import java.util.*;
class Data{
	private int x;
	private int y;
	public Data(int x, int y){
		this.x = x;
		this.y = y;
	}
}
public class closestpair{
	public Double dist(Data d1, Data d2){
		return Math.sqrt((d1.x - d2.x)*(d1.x - d2.x) + (d1.y - d2.y)*(d1.y - d2.y));
	}
	public Double strip(Data[] d, int size, Double d){
		Double min = d;
	}
