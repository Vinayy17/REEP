import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import axios from "axios";

const AuthContext = createContext();

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
const TOKEN_KEY = "token";
const USER_KEY = "user";
const EXPIRES_AT_KEY = "token_expires_at";

const clearStoredSession = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(EXPIRES_AT_KEY);
};

const setAxiosAuthHeader = (token) => {
  if (token) {
    axios.defaults.headers.common.Authorization = `Bearer ${token}`;
    return;
  }

  delete axios.defaults.headers.common.Authorization;
};

const getTokenExpiryFromJwt = (token) => {
  try {
    const [, payload] = token.split(".");
    if (!payload) return null;

    const normalized = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
    const decoded = JSON.parse(window.atob(padded));

    if (!decoded.exp) return null;

    return new Date(decoded.exp * 1000).toISOString();
  } catch (error) {
    return null;
  }
};

const getStoredSession = () => {
  const token = localStorage.getItem(TOKEN_KEY);
  const userData = localStorage.getItem(USER_KEY);
  const expiresAt = localStorage.getItem(EXPIRES_AT_KEY);

  if (!token || !userData) {
    return null;
  }

  return {
    token,
    user: JSON.parse(userData),
    expiresAt: expiresAt || getTokenExpiryFromJwt(token),
  };
};

const isSessionValid = (expiresAt) => {
  if (!expiresAt) return false;

  const expiryTime = new Date(expiresAt).getTime();
  return Number.isFinite(expiryTime) && expiryTime > Date.now();
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const logoutTimerRef = useRef(null);

  const clearLogoutTimer = useCallback(() => {
    if (logoutTimerRef.current) {
      window.clearTimeout(logoutTimerRef.current);
      logoutTimerRef.current = null;
    }
  }, []);

  const logout = useCallback(() => {
    clearLogoutTimer();
    clearStoredSession();
    setAxiosAuthHeader(null);
    setUser(null);
  }, [clearLogoutTimer]);

  const scheduleAutoLogout = useCallback((expiresAt) => {
    clearLogoutTimer();

    if (!isSessionValid(expiresAt)) {
      logout();
      return;
    }

    const timeout = new Date(expiresAt).getTime() - Date.now();
    logoutTimerRef.current = window.setTimeout(() => {
      logout();
    }, timeout);
  }, [clearLogoutTimer, logout]);

  const persistSession = useCallback(({ access_token, user: userData, expires_at }) => {
    const expiresAt = expires_at || getTokenExpiryFromJwt(access_token);

    localStorage.setItem(TOKEN_KEY, access_token);
    localStorage.setItem(USER_KEY, JSON.stringify(userData));

    if (expiresAt) {
      localStorage.setItem(EXPIRES_AT_KEY, expiresAt);
    } else {
      localStorage.removeItem(EXPIRES_AT_KEY);
    }

    setAxiosAuthHeader(access_token);
    setUser(userData);
    scheduleAutoLogout(expiresAt);

    return userData;
  }, [scheduleAutoLogout]);

  useEffect(() => {
    try {
      const session = getStoredSession();

      if (session && isSessionValid(session.expiresAt)) {
        setAxiosAuthHeader(session.token);
        setUser(session.user);
        scheduleAutoLogout(session.expiresAt);
      } else {
        logout();
      }
    } catch (error) {
      logout();
    } finally {
      setLoading(false);
    }

    return () => {
      clearLogoutTimer();
    };
  }, [logout, scheduleAutoLogout, clearLogoutTimer]);

  const login = async (email, password) => {
    const response = await axios.post(`${API}/auth/login`, { email, password });
    return persistSession(response.data);
  };

  const register = async (email, password, name) => {
    const response = await axios.post(`${API}/auth/register`, { email, password, name });
    return persistSession(response.data);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
