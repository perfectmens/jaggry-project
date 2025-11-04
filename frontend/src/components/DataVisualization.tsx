import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { parse, differenceInSeconds } from 'date-fns';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

// Define the structure of our batch data
interface BatchData {
  batch_id: number;
  start_time: string;
  end_time: string;
  next_batch_start: string;
}

const DataVisualization: React.FC = () => {
  const [data, setData] = useState<any[]>([]);
  const [filteredData, setFilteredData] = useState<any[]>([]);
  const [startDate, setStartDate] = useState<Date | null>(null);
  const [endDate, setEndDate] = useState<Date | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/batch-data');
        const transformedData = response.data.map((item: BatchData) => {
          const today = new Date().toISOString().split('T')[0];
          const start = parse(`${today}T${item.start_time}`);
          const end = parse(`${today}T${item.end_time}`);
          const duration = differenceInSeconds(end, start);
          return {
            name: `Batch ${item.batch_id}`,
            duration: duration > 0 ? duration : 0, // Handle overnight batches if necessary
            startTime: start
          };
        });
        setData(transformedData);
        setFilteredData(transformedData);
      } catch (error) {
        console.error('Error fetching batch data:', error);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    const applyFilter = () => {
        let filtered = data;
        if (startDate) {
            filtered = filtered.filter(item => item.startTime >= startDate);
        }
        if (endDate) {
            const endOfDay = new Date(endDate);
            endOfDay.setHours(23, 59, 59, 999);
            filtered = filtered.filter(item => item.startTime <= endOfDay);
        }
        setFilteredData(filtered);
    };
    applyFilter();
  }, [startDate, endDate, data]);

  return (
    <div className="bg-gray-800 rounded-lg p-4 h-full flex flex-col">
      <h3 className="text-xl font-semibold mb-4">Batch Duration Analysis (in seconds)</h3>

      {/* Date Range Picker */}
      <div className="flex justify-center mb-4 gap-4">
        <DatePicker
          selected={startDate}
          onChange={(date: Date | null) => setStartDate(date)}
          selectsStart
          startDate={startDate}
          endDate={endDate}
          placeholderText="Start Date"
          className="bg-gray-700 text-white p-2 rounded"
        />
        <DatePicker
          selected={endDate}
          onChange={(date: Date | null) => setEndDate(date)}
          selectsEnd
          startDate={startDate}
          endDate={endDate}
          minDate={startDate}
          placeholderText="End Date"
          className="bg-gray-700 text-white p-2 rounded"
        />
      </div>

      {/* Chart */}
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
            <XAxis dataKey="name" stroke="#cbd5e0" />
            <YAxis stroke="#cbd5e0" />
            <Tooltip contentStyle={{ backgroundColor: '#2d3748', border: '1px solid #4a5568' }} />
            <Legend />
            <Bar dataKey="duration" fill="#4299e1" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default DataVisualization;
