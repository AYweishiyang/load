package fun.shiyang.forecasting;

import java.sql.Date;
import java.sql.Timestamp;

/**
 * @author ay
 * @create 2021-05-09 20:52
 */
public class DeviceData {
    private String value;
    private Timestamp timestamp;

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public Timestamp getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Timestamp timestamp) {
        this.timestamp = timestamp;
    }
}
