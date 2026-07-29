export const Theme = {
  colors: {
    primary: '#1A73E8', // Google Blue
    primaryDark: '#174EA6',
    surface: '#FFFFFF', // Card background
    surfaceAlt: '#F1F3F4', // Slightly darker surface for active states
    background: '#F8F9FA', // Main background
    text: '#202124',
    textSecondary: '#5F6368',
    textHint: '#80868B',
    textInverse: '#FFFFFF', // Text on primary
    border: '#DADCE0',
    success: '#1E8E3E',
    successBg: '#E6F4EA',
    warning: '#F9AB00',
    warningBg: '#FEF7E0',
    error: '#D93025',
    errorBg: '#FCE8E6',
    overlay: 'rgba(32, 33, 36, 0.6)',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  radius: {
    sm: 8,
    md: 16,
    lg: 24,
    round: 9999,
  },
  typography: {
    h1: { fontSize: 24, fontWeight: '700' as const, color: '#202124' },
    h2: { fontSize: 20, fontWeight: '600' as const, color: '#202124' },
    subtitle: { fontSize: 16, fontWeight: '500' as const, color: '#5F6368' },
    body: { fontSize: 14, color: '#202124' },
    bodySecondary: { fontSize: 14, color: '#5F6368' },
    caption: { fontSize: 12, color: '#5F6368' },
    button: { fontSize: 14, fontWeight: '600' as const, letterSpacing: 0.5 },
  },
  shadows: {
    sm: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    md: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.15,
      shadowRadius: 6,
      elevation: 4,
    },
    lg: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.2,
      shadowRadius: 12,
      elevation: 8,
    },
  },
};
