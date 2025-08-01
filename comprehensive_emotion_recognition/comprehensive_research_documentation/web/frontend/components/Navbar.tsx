'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Brain, Database, Zap, Bot, TrendingUp, FileText, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

const Navbar = () => {
  const pathname = usePathname();
  
  const navItems = [
    { href: '/', label: 'Home', icon: Brain },
    { href: '/dataset', label: 'Dataset', icon: Database },
    { href: '/features', label: 'Features', icon: Zap },
    { href: '/models', label: 'Models', icon: Bot },
    { href: '/results', label: 'Results', icon: TrendingUp },
    { href: '/documentation', label: 'Docs', icon: FileText },
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">🧠</span>
              </div>
              <span className="font-bold text-xl text-gray-900">EEG Emotion AI</span>
            </Link>
          </div>
          
          <div className="flex items-center space-x-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-link ${router?.pathname === item.href ? 'active' : ''}`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
