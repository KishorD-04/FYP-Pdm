document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('sensorForm');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form values
        const formData = {
            voltage: document.getElementById('voltage').value,
            current: document.getElementById('current').value,
            temperature: document.getElementById('temperature').value,
            power: document.getElementById('power').value,
            humidity: document.getElementById('humidity').value,
            vibration: document.getElementById('vibration').value,
            ambient_temp: document.getElementById('ambient_temp').value,
            ambient_humidity: document.getElementById('ambient_humidity').value,
            x_axis: document.getElementById('x_axis').value,
            y_axis: document.getElementById('y_axis').value,
            z_axis: document.getElementById('z_axis').value
        };
        
        // Update displayed sensor values
        updateSensorDisplay(formData);
        
        // Make prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Update results
            document.getElementById('maintenance-needed').textContent = data.maintenance_needed;
            document.getElementById('maintenance-type').textContent = data.maintenance_type;
            document.getElementById('fault-detected').textContent = data.fault_detected ? 'Yes' : 'No';
            document.getElementById('repair-time').textContent = data.repair_time_hrs > 0 ? 
                `${data.repair_time_hrs} hours` : 'None';
            
            // Highlight cards based on results
            highlightResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while making the prediction');
        });
    });
    
    function updateSensorDisplay(data) {
        document.getElementById('display-voltage').textContent = `${data.voltage} V`;
        document.getElementById('display-current').textContent = `${data.current} A`;
        document.getElementById('display-temperature').textContent = `${data.temperature} °C`;
        document.getElementById('display-power').textContent = `${data.power} W`;
        document.getElementById('display-humidity').textContent = `${data.humidity} %`;
        document.getElementById('display-vibration').textContent = `${data.vibration} m/s²`;
    }
    
    function highlightResults(data) {
        const maintenanceCard = document.querySelector('.maintenance-card');
        const faultCard = document.querySelector('.fault-card');
        
        // Reset all animations
        maintenanceCard.style.animation = 'none';
        faultCard.style.animation = 'none';
        
        // Trigger animations based on results
        if (data.maintenance_needed === 'Yes') {
            maintenanceCard.style.animation = 'pulse-warning 2s infinite';
        }
        
        if (data.fault_detected) {
            faultCard.style.animation = 'pulse-danger 2s infinite';
        }
    }
});