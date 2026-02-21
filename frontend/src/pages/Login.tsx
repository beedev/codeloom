import { useState } from 'react';
import type { FormEvent } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.tsx';

export function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberDevice, setRememberDevice] = useState(false);
  const [error, setError] = useState('');
  const { login, isLoading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // Get the redirect path from location state, default to home
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/';

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!username.trim() || !password.trim()) {
      setError('Please enter both username and password');
      return;
    }

    const success = await login({ username, password });

    if (success) {
      navigate(from, { replace: true });
    } else {
      setError('Invalid username or password');
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-void">
      {/* Subtle radial gradient background */}
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_center,rgba(28,96,242,0.08)_0%,transparent_70%)]" />

      <div className="relative w-full max-w-md p-8 space-y-8">
        {/* Logo/Title */}
        <div className="flex flex-col items-center text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-glow/10">
            <span className="material-symbols-outlined text-[28px] text-glow">auto_awesome</span>
          </div>
          <h1 className="mt-4 text-2xl font-bold text-text">CodeLoom</h1>
          <p className="mt-2 text-sm text-text-muted">Code Intelligence & Migration Platform</p>
        </div>

        {/* Login Form — glassmorphic card */}
        <form
          onSubmit={handleSubmit}
          className="space-y-5 rounded-xl border border-void-surface bg-void-light/60 p-6 shadow-soft backdrop-blur-md"
        >
          {/* Error Message */}
          {error && (
            <div className="rounded-lg border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
              {error}
            </div>
          )}

          {/* Username Field */}
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-text-muted">
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="mt-1 w-full rounded-lg border border-void-surface bg-void px-4 py-3 text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
              placeholder="Enter your username"
              autoComplete="username"
              disabled={isLoading}
            />
          </div>

          {/* Password Field */}
          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="password" className="block text-sm font-medium text-text-muted">
                Password
              </label>
              <button
                type="button"
                className="text-xs text-glow hover:text-glow-bright"
                tabIndex={-1}
              >
                Forgot?
              </button>
            </div>
            <div className="relative mt-1">
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-void-surface bg-void px-4 py-3 pr-10 text-text placeholder-text-dim focus:border-glow/50 focus:outline-none focus:ring-1 focus:ring-glow/50"
                placeholder="Enter your password"
                autoComplete="current-password"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-dim hover:text-text-muted"
                tabIndex={-1}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                <span className="material-symbols-outlined text-[18px]">
                  {showPassword ? 'visibility_off' : 'visibility'}
                </span>
              </button>
            </div>
          </div>

          {/* Remember this device */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={rememberDevice}
              onChange={(e) => setRememberDevice(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-void-surface bg-void text-glow focus:ring-glow/50"
            />
            <span className="text-xs text-text-muted">Remember this device</span>
          </label>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-glow py-3 px-4 font-medium text-white transition-colors hover:bg-glow-dim disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-glow/50 focus:ring-offset-2 focus:ring-offset-void"
          >
            {isLoading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Signing in...
              </span>
            ) : (
              <>
                <span className="material-symbols-outlined text-[18px]">lock</span>
                Sign In
              </>
            )}
          </button>
        </form>

        {/* Footer */}
        <div className="space-y-2 text-center">
          <p className="text-sm text-text-dim">
            Default: <span className="text-text-muted">admin</span> / <span className="text-text-muted">admin123</span>
          </p>
          <p className="text-[10px] font-medium uppercase tracking-widest text-text-dim/50">
            Secure Access
          </p>
          <p className="text-[10px] text-text-dim/40">
            Powered by AST + ASG + RAG
          </p>
        </div>
      </div>
    </div>
  );
}
