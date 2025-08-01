import React from 'react';
import Navbar from './Navbar';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                EEG-Based Emotion Recognition Research
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                SEED-IV Dataset • 97.7% Accuracy Achievement • Six-Stage Architecture
              </p>
            </div>
            <div className="flex space-x-6">
              <a href="#" className="text-sm text-gray-500 hover:text-gray-900">
                📊 Dataset
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-gray-900">
                📚 Documentation
              </a>
              <a href="#" className="text-sm text-gray-500 hover:text-gray-900">
                💻 Source Code
              </a>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-gray-200">
            <div className="flex justify-between items-center">
              <p className="text-xs text-gray-400">
                © 2025 EEG Research Team. Built with Next.js and Tailwind CSS.
              </p>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  <span>Stage 2 Complete: 97.7% Accuracy</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
