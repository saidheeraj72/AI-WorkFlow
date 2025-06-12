// src/components/Sidebar.jsx
import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import DocumentManager from './DocumentManager';
import './Sidebar.css';

const Sidebar = ({
  currentView,
  setCurrentView,
  sidebarSection,
  setSidebarSection,
  setMessages,
  setSelectedModel,
  setCurrentSessionId,
  currentSessionId,
  selectedDocumentIds,
  setSelectedDocumentIds,
  selectedFolderIds,
  setSelectedFolderIds,
  collapsed,
  setCollapsed,
  width,
  setWidth,
  isMobile
}) => {
  const { user, logout, isAdmin } = useAuth();
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingChats, setIsLoadingChats] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const [showDocumentManager, setShowDocumentManager] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  
  const sidebarRef = useRef(null);
  const dropdownRef = useRef(null);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowProfileDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Drag functionality
  const handleMouseDown = (e) => {
    if (collapsed || isMobile) return;
    setIsDragging(true);
    dragStartX.current = e.clientX;
    dragStartWidth.current = width;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    e.preventDefault();
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const deltaX = e.clientX - dragStartX.current;
    const newWidth = Math.max(200, Math.min(500, dragStartWidth.current + deltaX));
    setWidth(newWidth);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  useEffect(() => {
    const fetchChatHistory = async () => {
      setIsLoadingChats(true);
      try {
        const response = await axios.get('http://localhost:8000/sessions');
        setChatHistory(response.data);
      } catch (error) {
        console.error('Error fetching chat history:', error);
      } finally {
        setIsLoadingChats(false);
      }
    };

    fetchChatHistory();
  }, [currentSessionId]);

  const handleNewChat = () => {
    setCurrentView('chat');
    setSidebarSection('chats');
    setMessages([]);
    setCurrentSessionId(null);
    setSelectedModel('llama3-70b-8192');
    if (isMobile) setCollapsed(true);
  };

  const handleDeleteSession = async (sessionId, e) => {
    e.stopPropagation();

    if (!confirm('Are you sure you want to delete this chat? This will also delete all personal documents uploaded in this chat.')) {
      return;
    }

    try {
      await axios.delete(`http://localhost:8000/sessions/${sessionId}`);
      setChatHistory((prev) => prev.filter((session) => session.id !== sessionId));

      if (currentSessionId === sessionId) {
        setMessages([]);
        setCurrentSessionId(null);
        setCurrentView('home');
      }
    } catch (error) {
      console.error('Error deleting session:', error);
      alert('Failed to delete chat. Please try again.');
    }
  };

  const loadSession = async (sessionId) => {
    try {
      const response = await axios.get(`http://localhost:8000/sessions/${sessionId}`);
      setMessages(response.data.messages);
      setSelectedModel(response.data.model);
      setCurrentSessionId(sessionId);
      setCurrentView('chat');
      setSidebarSection('chats');
      if (isMobile) setCollapsed(true);
    } catch (error) {
      console.error('Error loading session:', error);
      alert('Failed to load chat. Please try again.');
    }
  };

  const handleLogout = () => {
    setShowProfileDropdown(false);
    logout();
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString([], {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return 'Unknown date';
    }
  };

  const totalSelectedItems = (selectedDocumentIds?.length || 0) + (selectedFolderIds?.length || 0);

  return (
    <>
      <div 
        ref={sidebarRef}
        className={`sidebar ${collapsed ? 'collapsed' : ''} ${isMobile ? 'mobile' : ''}`}
        style={{ width: collapsed ? (isMobile ? '0' : '60px') : `${width}px` }}
      >
        {/* Collapse Toggle Button */}
        <button 
          className="collapse-toggle"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {collapsed ? '→' : '←'}
        </button>

        {!collapsed && (
          <>
            <div className="sidebar-header">
              <div className="profile-section" ref={dropdownRef}>
                <div 
                  className="profile-icon-container"
                  onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                >
                  <div className="profile-icon">
                    <span>👤</span>
                  </div>
                  <div className="user-info">
                    <span className="email">{user?.email}</span>
                    <span className={`role-badge ${user?.role}`}>
                      {user?.role === 'admin' ? '👑 Admin' : '👤 User'}
                    </span>
                  </div>
                  <span className={`dropdown-arrow ${showProfileDropdown ? 'rotated' : ''}`}>▼</span>
                </div>
                
                {showProfileDropdown && (
                  <div className="profile-dropdown">
                    <div className="dropdown-item profile-info">
                      <strong>{user?.email}</strong>
                      <small>{user?.role === 'admin' ? '👑 Administrator' : '👤 User'}</small>
                    </div>
                    <div className="dropdown-divider"></div>
                    <button className="dropdown-item logout-btn" onClick={handleLogout}>
                      🚪 Logout
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="personal-space">
              <h3>Personal Space</h3>
            </div>

            <div className="nav-items">
              <div
                className={`nav-item ${currentView === 'home' ? 'active' : ''}`}
                onClick={() => setCurrentView('home')}
              >
                <span className="nav-icon">🏠</span>
                Home
              </div>
              <div className="nav-item" onClick={handleNewChat}>
                <span className="nav-icon">💬</span>
                New Chat
              </div>
            </div>

            {/* RAG Status Summary */}
            <div className="rag-summary-section">
              <div className="rag-summary-card">
                <div className="rag-summary-header">
                  <span className="rag-icon">
                    {totalSelectedItems > 0 ? '🔍' : '💭'}
                  </span>
                  <div className="rag-summary-text">
                    <h4>Knowledge Base</h4>
                    <p>
                      {totalSelectedItems > 0 
                        ? `${totalSelectedItems} items selected for RAG`
                        : 'General chat mode'
                      }
                    </p>
                  </div>
                </div>
                {totalSelectedItems > 0 && (
                  <div className="rag-breakdown">
                    {selectedFolderIds?.length > 0 && (
                      <small>📁 {selectedFolderIds.length} folder(s)</small>
                    )}
                    {selectedDocumentIds?.length > 0 && (
                      <small>📄 {selectedDocumentIds.length} document(s)</small>
                    )}
                  </div>
                )}
                <small className="rag-hint">
                  Selection available in chat area
                </small>
              </div>

              {/* Admin Document Manager Access */}
              {isAdmin() && (
                <div className="admin-tools">
                  <button
                    className="manage-documents-btn"
                    onClick={() => setShowDocumentManager(true)}
                    title="Manage Documents & Folders"
                  >
                    📁 Manage Documents
                  </button>
                </div>
              )}
            </div>

            <div className="scrollable-content">
              {/* Chat History Section */}
              {/* Chat History Section */}
              {sidebarSection !== 'documents' && (
                <div className="chat-history">
                  <h4>CHAT HISTORY</h4>
                  <div className="chat-history-list">
                    {isLoadingChats ? (
                      <div className="loading-state">
                        <p>Loading chats...</p>
                      </div>
                    ) : chatHistory.length === 0 ? (
                      <div className="no-items">
                        <p>No chat history yet</p>
                        <small>Start a new chat to begin</small>
                      </div>
                    ) : (
                      chatHistory.map((session) => (
                        <div
                          key={session.id}
                          className={`chat-item ${currentSessionId === session.id ? 'active' : ''}`}
                          onClick={() => loadSession(session.id)}
                        >
                          <span className="chat-icon">💬</span>
                          <div className="chat-details">
                            <span className="chat-title" title={session.title}>
                              {session.title}
                            </span>
                            <small className="message-count">{session.message_count} messages</small>
                          </div>
                          <button
                            className="delete-btn"
                            onClick={(e) => handleDeleteSession(session.id, e)}
                            title="Delete chat and personal documents"
                            aria-label="Delete chat and personal documents"
                          >
                            🗑️
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Resize Handle */}
        {!collapsed && !isMobile && (
          <div 
            className="resize-handle"
            onMouseDown={handleMouseDown}
            style={{ cursor: isDragging ? 'col-resize' : 'col-resize' }}
          />
        )}
      </div>

      {/* Document Manager Modal */}
      {showDocumentManager && (
        <DocumentManager
          onDocumentSelect={() => {}}
          selectedDocumentIds={[]}
          onClose={() => setShowDocumentManager(false)}
          isAdmin={isAdmin()}
        />
      )}
    </>
  );
};

export default Sidebar;
