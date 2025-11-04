import React from 'react';
import Chat from './components/Chat';
import DataVisualization from './components/DataVisualization';

function App() {
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-800 p-4 flex-shrink-0">
        <h1 className="text-2xl font-bold mb-6">RAG Dashboard</h1>
        <nav>
          <ul>
            <li className="mb-2">
              <a href="#" className="block p-2 rounded bg-gray-700">Dashboard</a>
            </li>
          </ul>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 flex flex-col min-w-0">
        <h2 className="text-3xl font-semibold mb-6">Analytics Dashboard</h2>
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Left Column: Chat */}
          <div className="flex flex-col">
            <h3 className="text-xl font-semibold mb-4">LLM Chat Assistant</h3>
            <div className="flex-1">
              <Chat />
            </div>
          </div>

          {/* Right Column: Data Visualization */}
          <div className="flex flex-col">
            <DataVisualization />
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
