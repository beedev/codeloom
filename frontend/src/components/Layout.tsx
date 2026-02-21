/**
 * Layout Component
 *
 * App shell with a sticky top nav bar and main content area.
 * Nav: Logo + permanent links (Dashboard, Migration Plans, Settings),
 * Right: search bar, notifications bell, user avatar with logout dropdown.
 */

import { useState, useRef, useEffect } from 'react';
import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LogOut } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext.tsx';

interface LayoutProps {
  children: ReactNode;
  projects?: unknown[];
  projectsLoading?: boolean;
}

export function Layout({ children }: LayoutProps) {
  const { user, logout } = useAuth();
  const location = useLocation();

  const [showUserMenu, setShowUserMenu] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);

  // Close user menu on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setShowUserMenu(false);
      }
    }
    if (showUserMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showUserMenu]);

  // Get user initial for avatar
  const userInitial = (user?.username ?? 'U')[0].toUpperCase();

  return (
    <div className="flex h-screen flex-col bg-void text-text">
      {/* Top Nav Bar */}
      <header className="sticky top-0 z-50 flex h-14 shrink-0 items-center justify-between border-b border-void-surface bg-void-light/80 px-4 backdrop-blur-md">
        {/* Left: Logo + Nav Links */}
        <div className="flex items-center gap-6">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 text-text">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-glow/10">
              <span className="material-symbols-outlined text-[18px] text-glow">auto_awesome</span>
            </div>
            <span className="text-sm font-semibold">CodeLoom</span>
          </Link>

          {/* Permanent Nav Links */}
          <nav className="flex items-center gap-1">
            <NavLink
              to="/"
              icon={<span className="material-symbols-outlined text-[16px]">dashboard</span>}
              label="Dashboard"
              active={location.pathname === '/'}
            />
            <NavLink
              to="/migrations"
              icon={<span className="material-symbols-outlined text-[16px]">swap_horiz</span>}
              label="Migration Plans"
              active={location.pathname === '/migrations' || location.pathname.startsWith('/migration/')}
            />
            <NavLink
              to="/settings"
              icon={<span className="material-symbols-outlined text-[16px]">settings</span>}
              label="Settings"
              active={location.pathname === '/settings'}
            />
          </nav>
        </div>

        {/* Right: Search + Notifications + Avatar */}
        <div className="flex items-center gap-3">
          {/* Search bar */}
          <div className="relative">
            <span className="material-symbols-outlined absolute left-2.5 top-1/2 -translate-y-1/2 text-[16px] text-text-dim">
              search
            </span>
            <input
              type="text"
              placeholder="Search resources..."
              className="h-8 w-48 rounded-lg border border-void-surface bg-void pl-8 pr-3 text-xs text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
            />
          </div>

          {/* Notifications bell */}
          <button
            className="relative rounded-md p-1.5 text-text-dim transition-colors hover:bg-void-surface hover:text-text"
            aria-label="Notifications"
          >
            <span className="material-symbols-outlined text-[18px]">notifications</span>
          </button>

          {/* User avatar with dropdown */}
          <div className="relative" ref={userMenuRef}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-glow text-xs font-bold text-white transition-opacity hover:opacity-90"
              aria-label="User menu"
            >
              {userInitial}
            </button>

            {showUserMenu && (
              <div className="absolute right-0 top-full z-50 mt-1 w-44 rounded-lg border border-void-surface bg-void-light shadow-soft">
                <div className="border-b border-void-surface px-3 py-2">
                  <p className="text-xs font-medium text-text">{user?.username ?? 'User'}</p>
                  <p className="text-[10px] text-text-dim">Signed in</p>
                </div>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    logout();
                  }}
                  className="flex w-full items-center gap-2 px-3 py-2 text-xs text-text-muted transition-colors hover:bg-void-surface hover:text-danger"
                >
                  <LogOut className="h-3.5 w-3.5" />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-1 flex-col overflow-hidden">{children}</main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NavLink sub-component
// ---------------------------------------------------------------------------

function NavLink({
  to,
  icon,
  label,
  active,
}: {
  to: string;
  icon: ReactNode;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      to={to}
      className={`flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
        active
          ? 'bg-glow/10 text-glow border border-glow/20'
          : 'text-text-muted hover:bg-void-surface hover:text-text'
      }`}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}
