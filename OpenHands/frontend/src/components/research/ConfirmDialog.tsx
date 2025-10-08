import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, AlertTriangle, Info } from 'lucide-react';

interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'danger' | 'warning' | 'info';
  isLoading?: boolean;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'danger',
  isLoading = false,
}) => {
  const dialogRef = useRef<HTMLDivElement>(null);
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isOpen) {
      confirmButtonRef.current?.focus();

      const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Escape') {
          onClose();
        } else if (e.key === 'Enter') {
          onConfirm();
        } else if (e.key === 'Tab') {
          // Basic focus trap: cycle between cancel and confirm buttons
          const activeElement = document.activeElement;
          if (activeElement === confirmButtonRef.current && !e.shiftKey) {
            e.preventDefault();
            cancelButtonRef.current?.focus();
          } else if (activeElement === cancelButtonRef.current && e.shiftKey) {
            e.preventDefault();
            confirmButtonRef.current?.focus();
          }
        }
      };

      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, onClose, onConfirm]);

  const getIcon = () => {
    switch (variant) {
      case 'warning':
        return <AlertTriangle className="w-6 h-6 text-yellow-500" />;
      case 'info':
        return <Info className="w-6 h-6 text-blue-500" />;
      default:
        return <AlertCircle className="w-6 h-6 text-red-500" />;
    }
  };

  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  };

  const dialogVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: { opacity: 1, scale: 1 },
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm"
          variants={backdropVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          onClick={onClose}
        >
          <motion.div
            className="bg-white dark:bg-slate-900 p-6 rounded-lg shadow-lg max-w-md w-full mx-4"
            variants={dialogVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            onClick={(e) => e.stopPropagation()}
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="dialog-title"
            aria-describedby="dialog-message"
          >
            <div className="flex items-center mb-4">
              {getIcon()}
              <h2 id="dialog-title" className="ml-3 text-lg font-semibold text-gray-900 dark:text-slate-100">
                {title}
              </h2>
            </div>
            <p id="dialog-message" className="text-gray-700 dark:text-slate-300 mb-6">
              {message}
            </p>
            <div className="flex justify-end space-x-3">
              <button
                ref={cancelButtonRef}
                onClick={onClose}
                className="px-4 py-2 text-gray-700 dark:text-slate-300 bg-gray-200 dark:bg-slate-700 rounded hover:bg-gray-300 dark:hover:bg-slate-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                {cancelText}
              </button>
              <button
                ref={confirmButtonRef}
                onClick={onConfirm}
                disabled={isLoading}
                className={`px-4 py-2 rounded focus:outline-none focus:ring-2 ${
                  variant === 'danger'
                    ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500 text-white'
                    : variant === 'warning'
                    ? 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500 text-white'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isLoading ? 'Loading...' : confirmText}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ConfirmDialog;